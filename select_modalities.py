"""
Discover patients/studies under medulloblastoma/raw/, shortlist candidate
t1n/t1c/t2w/t2f series per study, and print a ready-to-edit CSV row skeleton
for medulloblastoma/subjects_registry.csv.

Adapted from ../clinical_exports/select_modalities.py for this export
format's two differences: no DICOMDIR to walk (series are discovered by
walking the filesystem directly -- see inventory_export.py's
find_series_roots()), and the series description is already in the folder
name (e.g. `SER_0006_T1 ax`), so no header read is needed just to know what a
series is *called*. A header read is still required to CONFIRM pre/post
contrast and plane, per the same discipline as ../MODALITY_SELECTION.md --
never trust a name (including this export's own "+c" suffix) over the
ContrastBolusAgent tag.

This does NOT pick the final SER numbers for you -- a human still checks
contrast, 3D-vs-2D, and orientation. What this automates is finding every
study not yet in the registry and printing the header detail needed to make
that call.

Usage:
    python3 select_modalities.py                       # scans medulloblastoma/raw/, only studies not yet in the registry
    python3 select_modalities.py raw/1-Medulloblastoma  # limit to one patient folder
    python3 select_modalities.py raw/1-Medulloblastoma/20230101  # limit to one study folder
    python3 select_modalities.py --all                  # re-show everything
    python3 select_modalities.py --json out.json
"""
import argparse
import csv
import re
from pathlib import Path

import pydicom

from inventory_export import find_series_roots, _series_files, SER_NAME_RE

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_ROOT = SCRIPT_DIR / "raw"
REGISTRY_PATH = SCRIPT_DIR / "subjects_registry.csv"

NON_IMAGE_MODALITIES = {"PR", "KO", "SR"}

# This scanner/export's own naming quirks, discovered while building this
# registry -- kept here rather than silently baked into the regexes so a
# future dataset with a genuinely different "APD"/"GRE" meaning doesn't get
# silently misclassified:
#   - APD  -- assumed to be an ADC-derived diffusion map (this export's own
#     abbreviation; not a standard DICOM term). CONFIRM from the header
#     (Modality/ImageType/SeriesDescription) before trusting this assumption
#     on a new dataset -- see ../MODALITY_SELECTION.md's approach to
#     per-protocol naming quirks (e.g. GE's "T1 FLAIR").
#   - GRE  -- gradient-echo / susceptibility-weighted series, not a BraTS
#     target modality here (same role as SWAN/SWI in ../PreOP,../PostOP).
#   - PD   -- proton-density, not a BraTS target modality.
EXCLUDE_PATTERN = re.compile(
    r"SCOUT|LOCALIZER|\bDWI\b|\bADC\b|\bAPD\b|SWI|SWAN|\bDTI\b|PERF|\bGRE\b|\bPD\b|\bMIP\b",
    re.IGNORECASE,
)

T1_PATTERN = re.compile(r"\bT1\b|MPRAGE|FSPGR|\bTFE\b", re.IGNORECASE)
T2_FLAIR_PATTERN = re.compile(r"\bT2\b|FLAIR|\bCUBE\b", re.IGNORECASE)

MIN_CANDIDATES_NOTE = (
    "no plausible T1-like and/or T2/FLAIR-like series found; not added to registry"
)


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slice_normal(orientation):
    if orientation is None:
        return None
    row = [float(v) for v in orientation[:3]]
    col = [float(v) for v in orientation[3:]]
    return [
        row[1] * col[2] - row[2] * col[1],
        row[2] * col[0] - row[0] * col[2],
        row[0] * col[1] - row[1] * col[0],
    ]


def _plane(orientation) -> str:
    normal = _slice_normal(orientation)
    if normal is None:
        return "?"
    return ["SAG", "COR", "AX"][max(range(3), key=lambda i: abs(normal[i]))]


def read_series_headers(series_dir: Path) -> dict:
    """Header detail from a series' first readable file -- enough to confirm
    contrast/plane/3D-vs-2D without reading every slice (unlike
    inventory_export.py, which needs every file for gap detection)."""
    files = _series_files(series_dir)
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(ds, "Rows", None) is None:
            return {"no_pixel_data": True, "modality": str(getattr(ds, "Modality", ""))}
        return {
            "modality": str(getattr(ds, "Modality", "")),
            "rows": getattr(ds, "Rows", None),
            "columns": getattr(ds, "Columns", None),
            "slice_thickness": _as_float(getattr(ds, "SliceThickness", None)),
            "contrast_agent": str(getattr(ds, "ContrastBolusAgent", "") or ""),
            "plane": _plane(getattr(ds, "ImageOrientationPatient", None)),
            "acquisition_type": str(getattr(ds, "MRAcquisitionType", "") or ""),
            "patient_id": str(getattr(ds, "PatientID", "")),
            "patient_name": str(getattr(ds, "PatientName", "")),
        }
    return {}


def classify(description: str, modality: str) -> str:
    if modality in NON_IMAGE_MODALITIES:
        return None
    if EXCLUDE_PATTERN.search(description):
        return None
    if T1_PATTERN.search(description):
        return "t1"
    if T2_FLAIR_PATTERN.search(description):
        return "t2f"
    return None


def format_candidate(ser_id: str, description: str, headers: dict, on_disk: int) -> str:
    contrast = headers.get("contrast_agent") or "none"
    plane = headers.get("plane", "?")
    dims = f"{headers.get('rows')}x{headers.get('columns')}"
    acq = headers.get("acquisition_type") or "?"
    thk = headers.get("slice_thickness")
    thk_str = f"{thk}mm" if thk is not None else "?"
    return (
        f"      {ser_id:>9}  {description:<20} files={on_disk:<5} "
        f"plane={plane:<4} dims={dims:<10} acq={acq:<2} thk={thk_str:<6} contrast={contrast}"
    )


def load_registered_keys(path: Path) -> set:
    """Set of (local_dir, study_date) triples already in the registry."""
    if not path.exists():
        return set()
    keys = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [(name or "").strip() for name in reader.fieldnames]
        for row in reader:
            local_dir = (row.get("local_dir") or "").strip()
            study_date = (row.get("study_date") or "").strip()
            if local_dir and study_date:
                keys.add((local_dir, study_date))
    return keys


def find_studies(root: Path) -> list:
    """Every (local_dir, study_date, series_root) at or under `root`, which
    may be medulloblastoma/raw/ itself, a single patient folder, or a single
    study folder. study_date is the first path component below the patient
    folder, so an arbitrarily-nested/duplicated wrapper folder (this export
    tool's zip layout sometimes repeats the date folder name) collapses to
    one stable label instead of leaking nesting depth into the registry.

    `local_dir` must always be the *patient* folder name for the registry's
    sake (`braTS_preprocess.py` later does `RAW_ROOT / local_dir` to relocate
    the study), so a narrower `root` under RAW_ROOT is resolved back to its
    patient folder there first, then results are filtered down to whatever
    `root` actually asked for.

    A `root` that lives entirely outside RAW_ROOT (e.g. the data was never
    copied into this checkout's raw/ -- pointing straight at wherever it
    actually sits on disk) has no such anchor, so it is assumed to BE the
    patient folder itself: `root.name` becomes `local_dir`. That only comes
    out right when `root` is a patient folder (or `raw/` was in play); a
    narrower *study* folder passed from outside raw/ has no way to recover
    its parent's real name and will wrongly use the study-date folder's own
    name as `local_dir` -- copy into raw/ first (or point one level up, at
    the patient folder) if you need a specific study picked out that way."""
    studies = []
    if not root.is_dir():
        return studies
    try:
        rel_to_raw = root.resolve().relative_to(RAW_ROOT.resolve())
        under_raw_root = True
    except ValueError:
        rel_to_raw = None
        under_raw_root = False

    if under_raw_root and not rel_to_raw.parts:
        for patient_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for series_root in find_series_roots(patient_dir):
                rel = series_root.relative_to(patient_dir)
                study_date = rel.parts[0] if rel.parts else patient_dir.name
                studies.append((patient_dir.name, study_date, series_root))
        return studies

    patient_dir = (RAW_ROOT / rel_to_raw.parts[0]) if under_raw_root else root
    for series_root in find_series_roots(patient_dir):
        if series_root != root and root not in series_root.parents:
            continue  # outside the narrower root the caller asked for
        rel = series_root.relative_to(patient_dir)
        study_date = rel.parts[0] if rel.parts else patient_dir.name
        studies.append((patient_dir.name, study_date, series_root))
    return studies


def process_study(local_dir: str, study_date: str, series_root: Path,
                   registered_keys: set, show_all: bool):
    key = (local_dir, study_date)
    if key in registered_keys and not show_all:
        return

    candidates = {"t1": [], "t2f": []}
    patient_id, patient_name = "", ""
    for ser_dir in sorted(series_root.glob("SER_*")):
        match = SER_NAME_RE.match(ser_dir.name)
        if not match:
            continue
        ser_id, description = match.groups()
        headers = read_series_headers(ser_dir)
        if not headers or headers.get("no_pixel_data"):
            continue
        patient_id = patient_id or headers.get("patient_id", "")
        patient_name = patient_name or headers.get("patient_name", "")
        bucket = classify(description, headers.get("modality", ""))
        if bucket:
            on_disk = len(_series_files(ser_dir))
            candidates[bucket].append((ser_id, description, headers, on_disk))

    print(f"\n=== {local_dir}  {study_date}  PatientID={patient_id} ({patient_name}) ===")
    if key in registered_keys:
        print("  (already in subjects_registry.csv)")

    if not candidates["t1"] or not candidates["t2f"]:
        print(f"  SKIPPING {local_dir}/{study_date} -- {MIN_CANDIDATES_NOTE}")
        return

    print("  T1-like candidates (t1n / t1c):")
    for ser_id, description, headers, on_disk in candidates["t1"]:
        print(format_candidate(ser_id, description, headers, on_disk))
    print("  T2/FLAIR-like candidates (t2w / t2f):")
    for ser_id, description, headers, on_disk in candidates["t2f"]:
        print(format_candidate(ser_id, description, headers, on_disk))

    print("\n  CSV row skeleton (fill subject_id + SER numbers by hand):")
    print(f"    TODO_MDB_ID, {patient_id}, {local_dir}, {study_date}, TODO, TODO, TODO, TODO")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, nargs="?", default=RAW_ROOT,
                         help="raw/, a patient dir, or a study dir to scan (default: medulloblastoma/raw/)")
    parser.add_argument("--all", action="store_true", help="also re-show studies already in the registry")
    parser.add_argument("--json", type=Path, default=None, help="also write the full candidate listing as JSON")
    args = parser.parse_args()

    studies = find_studies(args.path)
    if not studies:
        print(f"No studies found under {args.path} "
              f"(drop exports into medulloblastoma/raw/<patient>/... first)")
        return

    registered_keys = load_registered_keys(REGISTRY_PATH)
    for local_dir, study_date, series_root in studies:
        process_study(local_dir, study_date, series_root, registered_keys, args.all)

    if args.json:
        import json
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            [{"local_dir": l, "study_date": s, "series_root": str(r)} for l, s, r in studies],
            indent=2,
        ))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
