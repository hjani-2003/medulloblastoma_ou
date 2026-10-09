"""
Inventory a medulloblastoma raw export for truncation -- WITHOUT a DICOMDIR.

This export tool (unlike the IMediaExport format used by PreOP/PostOP/
clinical_exports) doesn't produce a DICOMDIR index, and series folders are
named directly from their description (e.g. `SER_0006_T1 ax`) rather than
needing a header read to find out what a series is. Both of those make
discovery easier -- but the missing DICOMDIR means there is no independent
"here's what should be here" record to compare against, which is what
../inventory_export.py normally uses to catch a silently-truncated download
(see ../APARSON_DATASET.md for why that check exists at all: the SER_xxxx
folder skeleton can be created in full even when the objects inside never
arrive).

Without an index, this script can only check SELF-CONSISTENCY of what is
actually on disk, two ways:
  1. InstanceNumber contiguity within each series (gaps are visible even
     with no reference count).
  2. Cross-checking against `ImagesInAcquisition`/`NumberOfSlices` DICOM
     tags, when the scanner happened to embed one in the file itself.

**This is strictly weaker than ../inventory_export.py's DICOMDIR check.** A
series that stopped early with no gap in what did arrive, and with neither
of those tags present, LOOKS COMPLETE here and will not be flagged --that
failure mode is only catchable with an external index. Treat "COMPLETE"
below as "no red flags found", not "verified against a manifest". If a
series looks clinically important and undersized, cross-check its file
count against the RIS/PACS report or scanner console if at all possible.

Usage:
    # a single study folder, a patient folder, or the whole raw/ root --
    # it recurses to find every folder that directly contains SER_* dirs
    python3 inventory_export.py raw
    python3 inventory_export.py raw/1-Medulloblastoma

    python3 inventory_export.py raw --missing-only
    python3 inventory_export.py raw --json inventory.json
"""
import argparse
import json
import re
from pathlib import Path

import pydicom

SER_NAME_RE = re.compile(r"^(SER_\d+)_(.+)$")


def find_series_roots(root: Path) -> list:
    """Every directory at or under `root` that directly contains one or more
    SER_* subdirectories -- i.e. a study/session folder, regardless of how
    many identically-named wrapper folders sit above it (this export tool's
    zip layout sometimes nests a study folder inside another folder with the
    exact same name)."""
    if any(root.glob("SER_*")):
        return [root]
    return sorted(
        d for d in root.rglob("*")
        if d.is_dir() and any(d.glob("SER_*"))
    )


def _series_files(series_dir: Path) -> list:
    """Image files under a SER_xxxx folder, whether they sit inside an
    OBJ_xxxx subfolder (same as PreOP/PostOP/clinical_exports) or directly in
    the SER_xxxx folder itself."""
    obj_files = [f for f in series_dir.glob("OBJ_*/*") if f.is_file()]
    if obj_files:
        return sorted(obj_files)
    return sorted(f for f in series_dir.iterdir() if f.is_file())


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gap_ranges(instances) -> list:
    """Missing InstanceNumbers between min and max, as [start, end] ranges."""
    have = set(instances)
    missing = [i for i in range(min(have), max(have) + 1) if i not in have]
    if not missing:
        return []
    ranges, start, previous = [], missing[0], missing[0]
    for value in missing[1:]:
        if value != previous + 1:
            ranges.append([start, previous])
            start = value
        previous = value
    ranges.append([start, previous])
    return ranges


def inspect_series(series_dir: Path) -> dict:
    files = _series_files(series_dir)
    match = SER_NAME_RE.match(series_dir.name)
    ser_id, description = (match.group(1), match.group(2)) if match else (series_dir.name, "")

    record = {
        "series_dir": series_dir.name,
        "series_id": ser_id,
        "description": description,
        "on_disk": len(files),
    }
    if not files:
        record["status"] = "EMPTY"
        return record

    instances, declared_totals, first = [], set(), None
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if first is None:
            first = ds
        if getattr(ds, "InstanceNumber", None) is not None:
            instances.append(int(ds.InstanceNumber))
        # Either tag, when present, is the scanner's own record of how many
        # images the acquisition was *supposed* to produce -- the closest
        # thing to a manifest available without a DICOMDIR.
        for tag in ("ImagesInAcquisition", "NumberOfSlices"):
            v = _as_float(getattr(ds, tag, None))
            if v:
                declared_totals.add(int(v))

    if first is None:
        record["status"] = "UNREADABLE"
        return record

    record["modality"] = str(getattr(first, "Modality", ""))
    record["description"] = record["description"] or str(getattr(first, "SeriesDescription", "") or "")

    gaps = _gap_ranges(instances) if instances else None
    record["instance_gaps"] = gaps

    if len(declared_totals) > 1:
        # Conflicting tags across files in the same series -- don't guess.
        record["declared_total"] = sorted(declared_totals)
        record["status"] = "AMBIGUOUS-TOTAL"
    elif declared_totals:
        expected = next(iter(declared_totals))
        record["declared_total"] = expected
        if record["on_disk"] < expected:
            record["status"] = "SHORT"
        elif gaps:
            record["status"] = "GAPS"
        else:
            record["status"] = "COMPLETE"
    elif gaps:
        record["status"] = "GAPS"
    else:
        record["status"] = "UNVERIFIED"  # no manifest tag, no gaps -- best effort only

    return record


def inventory(series_root: Path) -> dict:
    series = [inspect_series(d) for d in sorted(series_root.glob("SER_*"))]
    patient_id, patient_name = "", ""
    for s in series:
        d = series_root / s["series_dir"]
        files = _series_files(d)
        if not files:
            continue
        try:
            ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
        except Exception:
            continue
        patient_id = str(getattr(ds, "PatientID", ""))
        patient_name = str(getattr(ds, "PatientName", ""))
        break
    return {
        "series_root": str(series_root),
        "patient_id": patient_id,
        "patient_name": patient_name,
        "series": series,
    }


def print_report(report: dict, missing_only: bool = False):
    print("=" * 100)
    print(f"STUDY: {report['series_root']}")
    print(f"  Patient: {report['patient_name']}  ID={report['patient_id']}")
    shown = [s for s in report["series"] if not (missing_only and s["status"] in ("COMPLETE", "UNVERIFIED"))]
    if shown:
        print(f"\n    {'Series':<12} {'disk':>6}  {'Status':<16} Description")
    for s in shown:
        print(f"    {s['series_id']:<12} {s['on_disk']:>6}  {s['status']:<16} {s.get('description', '')}")
        if s.get("declared_total") is not None:
            print(f"                 declared total (ImagesInAcquisition/NumberOfSlices): {s['declared_total']}")
        gaps = s.get("instance_gaps")
        if gaps:
            shown_gaps = gaps[:8]
            suffix = f" (+{len(gaps) - len(shown_gaps)} more)" if len(gaps) > len(shown_gaps) else ""
            print(f"                 gaps in InstanceNumber: {shown_gaps}{suffix}")
    n_flagged = sum(1 for s in report["series"] if s["status"] not in ("COMPLETE", "UNVERIFIED"))
    print(f"\n  {len(report['series'])} series, {n_flagged} flagged (EMPTY/SHORT/GAPS/AMBIGUOUS-TOTAL/UNREADABLE)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="raw/, a patient dir, or a study dir")
    parser.add_argument("--missing-only", action="store_true", help="hide COMPLETE/UNVERIFIED series")
    parser.add_argument("--json", type=Path, default=None, help="also write full inventory as JSON")
    args = parser.parse_args()

    series_roots = find_series_roots(args.path)
    if not series_roots:
        parser.error(f"No SER_* folders found at or under {args.path}")

    reports = []
    for series_root in series_roots:
        report = inventory(series_root)
        reports.append(report)
        print_report(report, missing_only=args.missing_only)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
