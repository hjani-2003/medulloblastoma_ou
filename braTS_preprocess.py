"""
Preprocess medulloblastoma/raw/ into a BraTS-like layout: 4 modalities
(t1n, t1c, t2w, t2f), rigidly co-registered, resampled to isotropic 1mm
spacing, skull-stripped, then affine-registered into the shared cross-patient
atlas space.

This is a third sibling of ../braTS_preprocess.py and
../clinical_exports/braTS_preprocess.py -- a separate ecosystem rather than a
new phase/session of either, because these are a different patient
population (medulloblastoma) from a different export tool, with a different
on-disk shape:
  - no DICOMDIR (see inventory_export.py for what that costs the truncation
    check)
  - no PAT_xxxx/STD_0000 wrapper -- series sit directly under a study-date
    folder, discovered by walking the filesystem for folders that contain
    SER_* subdirs (find_series_roots(), reused from inventory_export.py)
  - the series description is already in the folder name
    (`SER_0006_T1 ax`), not just in the DICOM header
Everything below step 1 (registration, resampling, skull-stripping, atlas
registration) is identical in method to the other two pipelines -- only
discovery/conversion differs.

Subject IDs use an MDB_ prefix (MDB_0000, MDB_0001, ...) rather than SUBJ_,
deliberately a separate namespace from the PreOP/PostOP/clinical_exports
glioma cohort's SUBJ_ IDs -- these are a different tumor type/population and
are not expected to overlap, so keeping the prefixes visually distinct avoids
ever mixing the two cohorts up by ID alone.

This script only READS from medulloblastoma/raw/. All output is written
under medulloblastoma/PreProcessed/; the source DICOM is never modified. The
skull/atlas-registration step reuses the MNI152 template already built by
the project-root pipeline (../BraTS_preprocessed/atlas/), read-only, so
results share one coordinate space with the other two pipelines' output.

Recurring workflow for new data:
    1. Copy the new export into medulloblastoma/raw/<patient_dir>/... --
       nesting depth under the patient folder doesn't matter, as long as a
       SER_* folder eventually shows up somewhere underneath.
    2. python3 inventory_export.py raw    -- verify it downloaded completely
       (read the caveic at the top of that file: without a DICOMDIR, this is
       weaker than the other two pipelines' truncation check).
    3. python3 select_modalities.py       -- shortlist t1n/t1c/t2w/t2f
       candidates per study.
    4. Manually add the resulting row(s) to subjects_registry.csv.
    5. python3 braTS_preprocess.py --skip-hdbet --skip-atlas --subject MDB_0000
       -- sanity-check paths/registry before running the expensive stages.
    6. python3 braTS_preprocess.py

Usage:
    python3 braTS_preprocess.py                       # every registered job
    python3 braTS_preprocess.py --subject MDB_0000     # only this subject
    python3 braTS_preprocess.py --skip-hdbet           # stop before skull-stripping
    python3 braTS_preprocess.py --skip-atlas           # stop before atlas registration
    python3 braTS_preprocess.py --device cuda          # if a GPU is available (default: cpu)
    python3 braTS_preprocess.py --out-root /path/out   # write stage folders somewhere other than
                                                        # medulloblastoma/PreProcessed/
    python3 braTS_preprocess.py --registry other.csv   # use a registry other than the one next to
                                                        # this script
    python3 braTS_preprocess.py --raw-root /path/to/raw  # raw DICOM lives somewhere
                                                          # other than medulloblastoma/raw/
                                                          # next to this script (e.g. data
                                                          # was never copied in locally) --
                                                          # local_dir values in the registry
                                                          # must be patient-folder names
                                                          # directly under this root

Output layout (medulloblastoma/PreProcessed/<subject_id>/<stage>/<modality>.nii.gz):
    00_raw_nifti/         DICOM -> NIfTI, no other changes
    01_registered/        t1c/t2w/t2f rigidly registered onto the t1n grid
    02_resampled_1mm/     all 4 modalities resampled to isotropic 1x1x1mm
    03_final/              skull-stripped final volumes + brain_mask.nii.gz
    04_atlas_space/        all 4 modalities + brain_mask affine-registered
                            into the shared MNI152 atlas grid
"""
import argparse
import csv
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import pydicom
import SimpleITK as sitk

from inventory_export import find_series_roots, SER_NAME_RE

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_ROOT = SCRIPT_DIR / "raw"  # default only -- override with --raw-root
PREPROCESS_DIR = SCRIPT_DIR / "PreProcessed"
REGISTRY_PATH = SCRIPT_DIR / "subjects_registry.csv"

# Shared with the other two pipelines -- read-only, built there if missing.
# Relative to the project root (one level up from this script's medulloblastoma/
# folder), since this atlas is genuinely cross-pipeline, not colocated with
# this script the way raw/ and the registry are.
ATLAS_PATH = SCRIPT_DIR.parent / "BraTS_preprocessed" / "atlas" / "MNI152_T1_1mm_brain.nii.gz"

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def load_registry(path: Path) -> list:
    """Load subjects_registry.csv into a flat list of job dicts:
        {"subject_id", "patient_id", "local_dir", "study_date",
         "modalities": {"t1n": SER_xxxx, ...}}
    Fails loudly on malformed data rather than letting a bad row surface
    later as a confusing KeyError deep in the pipeline."""
    if not path.exists():
        raise FileNotFoundError(
            f"Subject registry not found: {path}\n"
            f"Expected a CSV with columns: subject_id,patient_id,local_dir,study_date,t1n,t1c,t2w,t2f"
        )

    required_cols = ["subject_id", "patient_id", "local_dir", "study_date", *MODALITIES]
    jobs = []
    subject_patient_ids = {}
    seen_keys = set()

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty (no header row); expected columns: {', '.join(required_cols)}")
        reader.fieldnames = [(name or "").strip() for name in reader.fieldnames]
        missing_cols = [c for c in required_cols if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"{path}: header is missing column(s): {', '.join(missing_cols)}")
        for row_num, row in enumerate(reader, start=2):
            for col in required_cols:
                if not (row.get(col) or "").strip():
                    raise ValueError(f"{path}:{row_num}: missing value for '{col}'")

            subject_id = row["subject_id"].strip()
            patient_id = row["patient_id"].strip()
            local_dir = row["local_dir"].strip()
            study_date = row["study_date"].strip()

            prior_patient_id = subject_patient_ids.setdefault(subject_id, patient_id)
            if prior_patient_id != patient_id:
                raise ValueError(
                    f"{path}:{row_num}: patient_id '{patient_id}' for {subject_id} "
                    f"conflicts with earlier row's patient_id '{prior_patient_id}'"
                )

            key = (local_dir, study_date, subject_id)
            if key in seen_keys:
                raise ValueError(f"{path}:{row_num}: duplicate row for {local_dir}/{study_date}/{subject_id}")
            seen_keys.add(key)

            jobs.append({
                "subject_id": subject_id,
                "patient_id": patient_id,
                "local_dir": local_dir,
                "study_date": study_date,
                "modalities": {mod: row[mod].strip() for mod in MODALITIES},
            })

    return jobs


def series_root_for(job: dict, raw_root: Path) -> Path:
    """The study-date folder holding this job's SER_* dirs, regardless of how
    many identically-named wrapper folders sit above it on disk."""
    patient_dir = raw_root / job["local_dir"]
    for series_root in find_series_roots(patient_dir):
        rel = series_root.relative_to(patient_dir)
        study_date = rel.parts[0] if rel.parts else patient_dir.name
        if study_date == job["study_date"]:
            return series_root
    raise FileNotFoundError(
        f"No SER_* folder found for {job['local_dir']}/{job['study_date']} under {patient_dir}"
    )


def resolve_series_dir(series_root: Path, ser_id: str) -> Path:
    """SER_xxxx (as stored in the registry) -> the actual SER_xxxx_<description>
    folder on disk."""
    if (series_root / ser_id).is_dir():
        return series_root / ser_id
    matches = list(series_root.glob(f"{ser_id}_*"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one folder matching '{ser_id}_*' under {series_root}, found {len(matches)}"
        )
    return matches[0]


def verify_patient_identity(job: dict, series_root: Path):
    """Reads PatientID off a file in this job's study and checks it against
    the registry, so a mismatched export fails loudly instead of silently
    mixing up patients."""
    any_ser = next(series_root.glob("SER_*"), None)
    if any_ser is None:
        raise FileNotFoundError(f"No series found under {series_root}")
    any_file = next((f for f in any_ser.rglob("*") if f.is_file()), None)
    if any_file is None:
        raise FileNotFoundError(f"No files found under {any_ser}")

    ds = pydicom.dcmread(str(any_file), force=True, stop_before_pixels=True)
    actual_id = str(getattr(ds, "PatientID", ""))
    expected_id = job["patient_id"]
    if actual_id != expected_id:
        raise ValueError(
            f"Patient identity mismatch for {job['subject_id']} in "
            f"{job['local_dir']}/{job['study_date']}: expected PatientID={expected_id}, "
            f"found {actual_id} in {any_file}"
        )


def stage_dir(job: dict, name: str) -> Path:
    d = PREPROCESS_DIR / job["subject_id"] / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def convert_series(series_dir: Path, out_path: Path) -> sitk.Image:
    """DICOM (SER_xxxx/OBJ_xxxx/IM_xxxx layout, confirmed identical to the
    other two pipelines for this export) -> NIfTI, via SimpleITK/GDCM."""
    files = sorted(f for f in series_dir.glob("OBJ_*/*") if f.is_file())
    if not files:
        raise FileNotFoundError(f"No files in {series_dir}")

    if len(files) == 1:
        image = sitk.ReadImage(str(files[0]))
    else:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir), recursive=True)
        if not dicom_names:
            dicom_names = [str(f) for f in files]
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(out_path))
    return image


def step1_convert(job: dict, raw_root: Path) -> dict:
    series_root = series_root_for(job, raw_root)
    verify_patient_identity(job, series_root)
    out_dir = stage_dir(job, "00_raw_nifti")
    paths = {}
    for mod, ser_id in job["modalities"].items():
        ser_dir = resolve_series_dir(series_root, ser_id)
        out_path = out_dir / f"{mod}.nii.gz"
        convert_series(ser_dir, out_path)
        paths[mod] = out_path
        print(f"    [convert] {mod} <- {ser_dir.name} -> {out_path}")
    return paths


def register_rigid(fixed: sitk.Image, moving: sitk.Image) -> sitk.Image:
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_transform, inPlace=False)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = reg.Execute(fixed_f, moving_f)
    metric = reg.GetMetricValue()

    resampled = sitk.Resample(
        moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID(),
    )
    return resampled, metric


def step2_register(job: dict, raw_paths: dict) -> dict:
    out_dir = stage_dir(job, "01_registered")
    fixed = sitk.ReadImage(str(raw_paths["t1n"]))

    out_paths = {"t1n": out_dir / "t1n.nii.gz"}
    sitk.WriteImage(fixed, str(out_paths["t1n"]))
    print(f"    [register] t1n is the fixed reference -> {out_paths['t1n']}")

    for mod in ["t1c", "t2w", "t2f"]:
        moving = sitk.ReadImage(str(raw_paths[mod]))
        registered, metric = register_rigid(fixed, moving)
        out_path = out_dir / f"{mod}.nii.gz"
        sitk.WriteImage(registered, str(out_path))
        out_paths[mod] = out_path
        print(f"    [register] {mod} -> t1n grid (final MMI metric={metric:.4f}) -> {out_path}")

    return out_paths


def resample_to_grid(image: sitk.Image, size, spacing, origin, direction) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size)
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(image)


def step3_resample_isotropic(job: dict, registered_paths: dict, spacing_mm=1.0) -> dict:
    out_dir = stage_dir(job, "02_resampled_1mm")

    t1 = sitk.ReadImage(str(registered_paths["t1n"]))
    orig_size = t1.GetSize()
    orig_spacing = t1.GetSpacing()
    new_spacing = (spacing_mm, spacing_mm, spacing_mm)
    new_size = [int(round(osz * ospc / spacing_mm)) for osz, ospc in zip(orig_size, orig_spacing)]

    out_paths = {}
    for mod, path in registered_paths.items():
        image = sitk.ReadImage(str(path))
        resampled = resample_to_grid(
            image, new_size, new_spacing, t1.GetOrigin(), t1.GetDirection(),
        )
        out_path = out_dir / f"{mod}.nii.gz"
        sitk.WriteImage(resampled, str(out_path))
        out_paths[mod] = out_path
    print(f"    [resample] all 4 modalities -> isotropic {spacing_mm}mm, size={new_size}")
    return out_paths


def step4_skullstrip(job: dict, resampled_paths: dict, device: str = "cpu") -> dict:
    out_dir = stage_dir(job, "03_final")
    hdbet_work = out_dir / "_hdbet_tmp"
    hdbet_work.mkdir(exist_ok=True)
    try:
        return _skullstrip_with_hdbet(resampled_paths, out_dir, hdbet_work, device)
    finally:
        # HD-BET failing (or not being installed) must not leave a half-filled temp dir behind
        shutil.rmtree(hdbet_work, ignore_errors=True)


def _skullstrip_with_hdbet(resampled_paths: dict, out_dir: Path, hdbet_work: Path, device: str) -> dict:
    t1_brain = hdbet_work / "t1n_brain.nii.gz"
    cmd = [
        "hd-bet", "-i", str(resampled_paths["t1n"]), "-o", str(t1_brain),
        "-device", device, "--save_bet_mask",
    ]
    if device == "cpu":
        cmd.append("--disable_tta")
    print(f"    [hd-bet] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    mask_candidates = [f for f in hdbet_work.glob("*.nii.gz") if f.resolve() != t1_brain.resolve()]
    if not mask_candidates:
        raise RuntimeError(f"HD-BET did not produce a mask file in {hdbet_work}")
    mask_path = mask_candidates[0]
    mask = sitk.ReadImage(str(mask_path))

    out_paths = {}
    for mod, path in resampled_paths.items():
        image = sitk.ReadImage(str(path))
        img_arr = sitk.GetArrayFromImage(image)
        mask_arr = sitk.GetArrayFromImage(mask) > 0
        stripped_arr = img_arr * mask_arr
        stripped = sitk.GetImageFromArray(stripped_arr.astype(img_arr.dtype))
        stripped.CopyInformation(image)
        out_path = out_dir / f"{mod}.nii.gz"
        sitk.WriteImage(stripped, str(out_path))
        out_paths[mod] = out_path
        print(f"    [skull-strip] {mod} -> {out_path}")

    mask_out = out_dir / "brain_mask.nii.gz"
    sitk.WriteImage(mask, str(mask_out))
    out_paths["brain_mask"] = mask_out
    return out_paths


def prepare_atlas_template() -> Path:
    """Build a skull-stripped MNI152 brain template (fetched once via nilearn,
    cached at ATLAS_PATH) -- shared with the other two pipelines, so all
    three land in the same coordinate space."""
    if ATLAS_PATH.exists():
        return ATLAS_PATH

    import numpy as np
    import nibabel as nib
    from nilearn import datasets

    ATLAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    icbm = datasets.fetch_icbm152_2009(verbose=0)
    t1 = nib.load(icbm.t1)
    mask = nib.load(icbm.mask)
    brain_arr = t1.get_fdata().astype(np.float32) * (mask.get_fdata() > 0)
    brain_img = nib.Nifti1Image(brain_arr, t1.affine, t1.header)
    nib.save(brain_img, str(ATLAS_PATH))
    print(f"    [atlas] built skull-stripped MNI152 template -> {ATLAS_PATH}")
    return ATLAS_PATH


def register_affine(fixed: sitk.Image, moving: sitk.Image):
    """12-DOF affine (not deformable) -- see ../ATLAS_README.md for why."""
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=300,
        gradientMagnitudeTolerance=1e-8,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_transform, inPlace=False)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = reg.Execute(fixed_f, moving_f)
    metric = reg.GetMetricValue()
    resampled = sitk.Resample(
        moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID(),
    )
    return resampled, final_transform, metric


def step5_atlas_register(job: dict, final_paths: dict, atlas_path: Path) -> dict:
    out_dir = stage_dir(job, "04_atlas_space")
    atlas = sitk.ReadImage(str(atlas_path))

    t1 = sitk.ReadImage(str(final_paths["t1n"]))
    t1_registered, transform, metric = register_affine(atlas, t1)
    out_paths = {"t1n": out_dir / "t1n.nii.gz"}
    sitk.WriteImage(t1_registered, str(out_paths["t1n"]))
    print(f"    [atlas] t1n -> MNI152 space (final MMI metric={metric:.4f})")

    for mod in ["t1c", "t2w", "t2f"]:
        moving = sitk.ReadImage(str(final_paths[mod]))
        resampled = sitk.Resample(moving, atlas, transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
        out_path = out_dir / f"{mod}.nii.gz"
        sitk.WriteImage(resampled, str(out_path))
        out_paths[mod] = out_path
        print(f"    [atlas] {mod} -> MNI152 space (reused t1n's transform) -> {out_path}")

    mask = sitk.ReadImage(str(final_paths["brain_mask"]))
    mask_resampled = sitk.Resample(mask, atlas, transform, sitk.sitkNearestNeighbor, 0, mask.GetPixelID())
    mask_out = out_dir / "brain_mask.nii.gz"
    sitk.WriteImage(mask_resampled, str(mask_out))
    out_paths["brain_mask"] = mask_out

    return out_paths


def run_job(job: dict, raw_root: Path, skip_hdbet: bool, skip_atlas: bool, device: str):
    print(f"\n=== {job['subject_id']} ({job['local_dir']}/{job['study_date']}) ===")
    raw_paths = step1_convert(job, raw_root)
    registered_paths = step2_register(job, raw_paths)
    resampled_paths = step3_resample_isotropic(job, registered_paths)
    if skip_hdbet:
        print("    [skip] --skip-hdbet set, stopping before skull-stripping")
        return
    final_paths = step4_skullstrip(job, resampled_paths, device=device)
    if skip_atlas:
        print("    [skip] --skip-atlas set, stopping before atlas registration")
        return
    atlas_path = prepare_atlas_template()
    step5_atlas_register(job, final_paths, atlas_path)


def main():
    global PREPROCESS_DIR  # stage_dir() reads this module-level; --out-root overrides it
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT,
                         help="where the raw DICOM lives, matching subjects_registry.csv's "
                              "local_dir values (default: medulloblastoma/raw/ next to this script)")
    parser.add_argument("--out-root", type=Path, default=PREPROCESS_DIR,
                         help="where the stage folders (00_raw_nifti ... 04_atlas_space) are written "
                              "(default: medulloblastoma/PreProcessed/ next to this script)")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH,
                         help="subjects registry CSV (default: subjects_registry.csv next to this script)")
    parser.add_argument("--skip-hdbet", action="store_true")
    parser.add_argument("--skip-atlas", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    PREPROCESS_DIR = args.out_root.resolve()

    try:
        jobs = load_registry(args.registry)
    except (OSError, ValueError, csv.Error) as e:
        print(f"ERROR: could not load registry: {e}", file=sys.stderr)
        sys.exit(2)
    if not jobs:
        print(f"ERROR: {args.registry} has no rows yet -- run select_modalities.py (or preprocess_gui.py) "
              f"and add rows before running this script.", file=sys.stderr)
        sys.exit(1)

    if args.subject:
        jobs = [j for j in jobs if j["subject_id"] == args.subject]

    if not jobs:
        print(f"ERROR: no registry rows match --subject {args.subject}", file=sys.stderr)
        sys.exit(1)

    failed = []
    for job in jobs:
        try:
            run_job(job, args.raw_root, args.skip_hdbet, args.skip_atlas, args.device)
        except Exception as e:  # one bad job must not stop the remaining ones
            traceback.print_exc()
            print(f"\n!! FAILED {job['subject_id']} ({job['local_dir']}/{job['study_date']}): "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            print(f"   partial output (if any) is under {PREPROCESS_DIR / job['subject_id']}; "
                  f"a re-run overwrites it", file=sys.stderr)
            failed.append(job["subject_id"])

    if failed:
        print(f"\n{len(failed)} of {len(jobs)} job(s) FAILED: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)
    print(f"\nDone. Output under: {PREPROCESS_DIR}")


if __name__ == "__main__":
    main()
