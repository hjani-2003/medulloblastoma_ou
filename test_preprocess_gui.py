"""
Tests for preprocess_gui.py and the error handling in braTS_preprocess.py.

Everything runs on a small synthetic DICOM export built in a temp dir (no real
patient data needed). Run from this folder, in the unet_venv environment:

    python3 -m unittest test_preprocess_gui -v

Tk tests need a display and are skipped without one. HD-BET itself is never
run (the tests either skip it or deliberately make it unavailable).
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

import preprocess_gui as g

HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "braTS_preprocess.py"

# (series number, description, ContrastBolusAgent, acquisition type, slices)
DEFAULT_SERIES = [
    (1, "SCOUT", "", "2D", 3),
    (2, "T1 ax", "", "2D", 12),
    (3, "T1 ax+c", "PROHANCE", "2D", 12),
    (4, "T1 MPRAGE", "", "3D", 16),
    (5, "T2 ax", "", "2D", 12),
    (6, "T2 FLAIR ax", "", "2D", 12),
    (7, "DWI ax", "", "2D", 12),
]
HEADER = ", ".join(g.REGISTRY_COLUMNS)
GOOD_ROW = "MDB_0000, MB001, 1-Medulloblastoma, 2.10.2010, SER_0004, SER_0003, SER_0005, SER_0006"


def write_series(ser_dir: Path, n: int, contrast: str, acq: str, pid: str, drop=()):
    """One synthetic MR series at ser_dir/OBJ_0001/IM_xxxx (slice indices in `drop` are left out)."""
    rng = np.random.default_rng(1)
    study_uid, series_uid = generate_uid(), generate_uid()
    zz, yy, xx = np.mgrid[0:n, 0:32, 0:32]
    vol = (np.exp(-(((zz - n / 2) / (n / 3)) ** 2 + ((yy - 16) / 10) ** 2 + ((xx - 16) / 10) ** 2)) * 800
           + rng.normal(0, 5, (n, 32, 32))).clip(0).astype(np.uint16)
    obj = ser_dir / "OBJ_0001"
    obj.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        if k in drop:
            continue
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.Modality, ds.PatientID, ds.PatientName = "MR", pid, "Test^Fake"
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.SOPInstanceUID, ds.SOPClassUID = meta.MediaStorageSOPInstanceUID, MRImageStorage
        ds.Rows = ds.Columns = 32
        ds.PixelSpacing, ds.SliceThickness = [1.0, 1.0], 1.0
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, float(k)]
        ds.InstanceNumber, ds.MRAcquisitionType = k + 1, acq
        if contrast:
            ds.ContrastBolusAgent = contrast
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit, ds.PixelRepresentation = 15, 0
        ds.PixelData = vol[k].tobytes()
        ds.save_as(obj / f"IM_{k:04d}", enforce_file_format=True)


def build_export(raw: Path, patients=(("1-Medulloblastoma", "MB001", "2.10.2010"),), series=DEFAULT_SERIES):
    """raw/<patient>/<date>/<date>/SER_xxxx_<desc>/... (doubled date folder, like the real export)."""
    for local_dir, pid, date in patients:
        study = raw / local_dir / date / date
        for num, desc, contrast, acq, n in series:
            write_series(study / f"SER_{num:04d}_{desc}", n, contrast, acq, pid)
    return raw


def study_root(raw: Path, local_dir="1-Medulloblastoma", date="2.10.2010") -> Path:
    return raw / local_dir / date / date


def alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def registry(self, text: str) -> Path:
        p = self.tmp / "registry.csv"
        p.write_text(text)
        return p


# ---------------------------------------------------------------- scanning

class ScanTests(TempDirCase):
    def setUp(self):
        super().setUp()
        self.raw = build_export(self.tmp / "raw")

    def test_discover_collapses_doubled_date_folder(self):
        raw_root, studies = g.discover_studies(self.raw)
        self.assertEqual(raw_root, self.raw.resolve())
        self.assertEqual([(a, b) for a, b, _ in studies], [("1-Medulloblastoma", "2.10.2010")])

    def test_discover_single_study_folder_uses_parent_as_raw_root(self):
        raw_root, studies = g.discover_studies(study_root(self.raw))
        self.assertEqual(raw_root, study_root(self.raw).parent.resolve())
        self.assertEqual(len(studies), 1)

    def test_discover_nothing(self):
        (self.tmp / "empty").mkdir()
        self.assertEqual(g.discover_studies(self.tmp / "empty")[1], [])

    def test_scan_and_suggestions(self):
        info = g.scan_study(study_root(self.raw))
        self.assertEqual(info["patient_id"], "MB001")
        by = {s["ser_id"]: s for s in info["series"]}
        self.assertTrue(by["SER_0001"]["excluded"] and by["SER_0007"]["excluded"])
        self.assertEqual(g.suggest_modalities(info["series"]),
                         {"t1n": "SER_0004", "t1c": "SER_0003", "t2w": "SER_0005", "t2f": "SER_0006"})

    def test_truncated_series_is_flagged(self):
        shutil.rmtree(study_root(self.raw) / "SER_0004_T1 MPRAGE")
        write_series(study_root(self.raw) / "SER_0004_T1 MPRAGE", 16, "", "3D", "MB001", drop={5, 6})
        by = {s["ser_id"]: s for s in g.scan_study(study_root(self.raw))["series"]}
        self.assertEqual(by["SER_0004"]["status"], "GAPS")

    def test_corrupt_series_is_kept_and_flagged_not_dropped(self):
        ser = study_root(self.raw) / "SER_0004_T1 MPRAGE"
        shutil.rmtree(ser)
        (ser / "OBJ_0001").mkdir(parents=True)
        (ser / "OBJ_0001" / "IM_0000").write_bytes(b"this is not dicom" * 50)
        by = {s["ser_id"]: s for s in g.scan_study(study_root(self.raw))["series"]}
        self.assertEqual(by["SER_0004"]["status"], "UNREADABLE")

    def test_empty_series_is_flagged(self):
        ser = study_root(self.raw) / "SER_0004_T1 MPRAGE"
        shutil.rmtree(ser)
        (ser / "OBJ_0001").mkdir(parents=True)
        by = {s["ser_id"]: s for s in g.scan_study(study_root(self.raw))["series"]}
        self.assertEqual(by["SER_0004"]["status"], "EMPTY")

    def test_exception_in_one_series_does_not_abort_scan(self):
        real = g.read_series_headers

        def flaky(ser_dir):
            if ser_dir.name.startswith("SER_0002"):
                raise PermissionError("denied")
            return real(ser_dir)

        with mock.patch.object(g, "read_series_headers", flaky):
            info = g.scan_study(study_root(self.raw))
        by = {s["ser_id"]: s for s in info["series"]}
        self.assertEqual(by["SER_0002"]["status"], "ERROR")
        self.assertIn("PermissionError", by["SER_0002"]["error"])
        self.assertEqual(len(info["series"]), 7)
        self.assertEqual(by["SER_0003"]["status"], "UNVERIFIED")  # the rest still scanned

    def test_unusable_series_are_never_suggested(self):
        ser = study_root(self.raw) / "SER_0004_T1 MPRAGE"
        shutil.rmtree(ser)
        (ser / "OBJ_0001").mkdir(parents=True)  # the best t1n candidate is now EMPTY
        sug = g.suggest_modalities(g.scan_study(study_root(self.raw))["series"])
        self.assertEqual(sug["t1n"], "SER_0002")  # falls back to the 2D pre-contrast T1


# ---------------------------------------------------------------- registry

class RegistryTests(TempDirCase):
    def row(self, **kw):
        base = dict(subject_id="MDB_0000", patient_id="MB001", local_dir="1-Medulloblastoma",
                    study_date="2.10.2010", t1n="SER_0004", t1c="SER_0003", t2w="SER_0005", t2f="SER_0006")
        return {**base, **kw}

    def test_missing_empty_and_header_only_are_empty_registries(self):
        self.assertEqual(g.read_registry_rows(self.tmp / "nope.csv"), [])
        self.assertEqual(g.read_registry_rows(self.registry("")), [])
        self.assertEqual(g.read_registry_rows(self.registry(HEADER + "\n")), [])

    def test_header_missing_a_column_is_a_clear_error(self):
        with self.assertRaisesRegex(g.RegistryError, "missing column.*t2f"):
            g.read_registry_rows(self.registry("subject_id, patient_id, local_dir, study_date, t1n, t1c, t2w\n"))

    def test_binary_garbage_is_a_registry_error(self):
        p = self.tmp / "registry.csv"
        p.write_bytes(b"\xff\xfe\x00\x81" * 100)
        with self.assertRaises(g.RegistryError):
            g.read_registry_rows(p)

    def test_short_rows_are_padded_not_keyerror(self):
        rows = g.read_registry_rows(self.registry(HEADER + "\nMDB_0000, MB001\n"))
        self.assertEqual(rows[0]["t2f"], "")
        self.assertEqual(g.next_subject_id(rows), "MDB_0001")

    def test_add_creates_file_with_header_and_matches_style(self):
        p = self.tmp / "new.csv"
        g.add_registry_row(p, self.row())
        self.assertEqual(p.read_text(), HEADER + "\n" + GOOD_ROW + "\n")

    def test_add_appends_and_keeps_backup(self):
        p = self.registry(HEADER + "\n" + GOOD_ROW + "\n")
        g.add_registry_row(p, self.row(subject_id="MDB_0001", patient_id="MB002", study_date="5.5.2011"))
        self.assertEqual(len(g.read_registry_rows(p)), 2)
        self.assertEqual(p.with_name("registry.csv.bak").read_text(), HEADER + "\n" + GOOD_ROW + "\n")
        self.assertFalse(p.with_name("registry.csv.tmp").exists())

    def test_add_to_file_without_trailing_newline(self):
        p = self.registry(HEADER + "\n" + GOOD_ROW)  # hand-edited, no final newline
        g.add_registry_row(p, self.row(subject_id="MDB_0001", patient_id="MB002", study_date="5.5.2011"))
        self.assertEqual(len(g.read_registry_rows(p)), 2)

    def test_rejects_bad_rows_and_leaves_file_untouched(self):
        p = self.registry(HEADER + "\n" + GOOD_ROW + "\n")
        before = p.read_text()
        bad = [
            self.row(),                                              # exact duplicate
            self.row(study_date="x", patient_id="OTHER"),            # same subject_id, different patient
            self.row(subject_id="MDB_0009", t1n="", ),               # empty modality
            self.row(subject_id="MDB_0009", t1c="SER_0004"),         # same series twice
            self.row(subject_id="../evil"),                          # would escape the output dir
            self.row(subject_id="a/b"),
            self.row(subject_id=".."),
            self.row(subject_id="MDB_9,9"),                          # comma would shift CSV columns
            self.row(subject_id="MDB_0009", patient_id='a"b'),
        ]
        for row in bad:
            with self.subTest(row=row), self.assertRaises(g.RegistryError):
                g.add_registry_row(p, row)
        self.assertEqual(p.read_text(), before)

    def test_refuses_to_append_when_column_order_differs(self):
        p = self.registry("patient_id, subject_id, local_dir, study_date, t1n, t1c, t2w, t2f\n")
        with self.assertRaisesRegex(g.RegistryError, "columns must be exactly"):
            g.add_registry_row(p, self.row())

    def test_unwritable_directory_is_a_registry_error(self):
        if os.geteuid() == 0:
            self.skipTest("root ignores directory permissions")
        d = self.tmp / "ro"
        d.mkdir()
        (d / "registry.csv").write_text(HEADER + "\n")
        d.chmod(0o500)
        self.addCleanup(d.chmod, 0o700)
        with self.assertRaises(g.RegistryError):
            g.add_registry_row(d / "registry.csv", self.row())

    def test_concurrent_adds_lose_no_rows(self):
        p = self.registry(HEADER + "\n")
        errors = []

        def add(i):
            try:
                g.add_registry_row(p, self.row(subject_id=f"MDB_{i:04d}", patient_id=f"P{i}"))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=add, args=(i,)) for i in range(12)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        ids = sorted(r["subject_id"] for r in g.read_registry_rows(p))
        self.assertEqual(ids, [f"MDB_{i:04d}" for i in range(12)])

    def test_concurrent_conflicting_adds_only_one_wins(self):
        # same subject_id, different patients: validation must see the other writer's row under the lock
        p = self.registry(HEADER + "\n")
        results = []

        def add(pid):
            try:
                g.add_registry_row(p, self.row(subject_id="MDB_0000", patient_id=pid, study_date=pid))
                results.append("ok")
            except g.RegistryError:
                results.append("rejected")

        threads = [threading.Thread(target=add, args=(f"P{i}",)) for i in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(sorted(results), ["ok"] + ["rejected"] * 5)
        self.assertEqual(len(g.read_registry_rows(p)), 1)

    def test_rows_written_are_accepted_by_the_pipeline_loader(self):
        try:
            import braTS_preprocess as pipeline  # imports SimpleITK
        except ImportError:
            self.skipTest("SimpleITK not available")
        p = self.registry(HEADER + "\n")
        g.add_registry_row(p, self.row())
        g.add_registry_row(p, self.row(subject_id="MDB_0001", patient_id="MB002", study_date="5.5.2011"))
        jobs = pipeline.load_registry(p)
        self.assertEqual([j["subject_id"] for j in jobs], ["MDB_0000", "MDB_0001"])
        self.assertEqual(jobs[0]["modalities"]["t2f"], "SER_0006")


# ---------------------------------------------------------------- preflight

class PreflightTests(TempDirCase):
    def setUp(self):
        super().setUp()
        self.raw = build_export(self.tmp / "raw")
        self.out = self.tmp / "out"
        self.rows = g.read_registry_rows(self.registry(HEADER + "\n" + GOOD_ROW + "\n"))

    def run_pf(self, rows=None, raw=None, out=None, **kw):
        return g.preflight(self.rows if rows is None else rows, self.raw if raw is None else raw,
                           self.out if out is None else out, **kw)

    def test_valid_setup_is_clean(self):
        errors, warnings = self.run_pf()
        self.assertEqual((errors, warnings), ([], []))

    def test_output_inside_raw_is_an_error(self):
        errors, _ = self.run_pf(out=self.raw / "results")
        self.assertTrue(any("inside the raw folder" in e for e in errors), errors)
        errors, _ = self.run_pf(out=self.raw)
        self.assertTrue(any("inside the raw folder" in e for e in errors), errors)

    def test_output_path_that_is_a_file(self):
        f = self.tmp / "afile"
        f.write_text("x")
        errors, _ = self.run_pf(out=f / "sub")
        self.assertTrue(any("not a directory" in e for e in errors), errors)

    def test_unwritable_output(self):
        if os.geteuid() == 0:
            self.skipTest("root ignores directory permissions")
        ro = self.tmp / "ro"
        ro.mkdir()
        ro.chmod(0o500)
        self.addCleanup(ro.chmod, 0o700)
        errors, _ = self.run_pf(out=ro / "new")
        self.assertTrue(any("no permission" in e for e in errors), errors)

    def test_low_disk_space_warns(self):
        Usage = namedtuple("Usage", "total used free")
        with mock.patch.object(g.shutil, "disk_usage", return_value=Usage(100 * 2**30, 99 * 2**30, 2**30)):
            errors, warnings = self.run_pf()
        self.assertEqual(errors, [])
        self.assertTrue(any("GB free" in w for w in warnings), warnings)

    def test_missing_raw_root(self):
        errors, _ = self.run_pf(raw=self.tmp / "nowhere")
        self.assertTrue(any("raw root is not a folder" in e for e in errors), errors)

    def test_registry_row_pointing_at_missing_data(self):
        for field, value, text in (("local_dir", "9-Nope", "patient folder not found"),
                                   ("study_date", "1.1.1999", "no SER_\\* folders"),
                                   ("t1n", "SER_0099", "found 0")):
            with self.subTest(field=field):
                errors, _ = self.run_pf(rows=[{**self.rows[0], field: value}])
                self.assertTrue(any(__import__("re").search(text, e) for e in errors), errors)

    def test_registry_row_with_empty_field(self):
        errors, _ = self.run_pf(rows=[{**self.rows[0], "t2f": ""}])
        self.assertTrue(any("empty field" in e and "t2f" in e for e in errors), errors)

    def test_series_without_files(self):
        for f in (study_root(self.raw) / "SER_0006_T2 FLAIR ax" / "OBJ_0001").iterdir():
            f.unlink()
        errors, _ = self.run_pf()
        self.assertTrue(any("no image files" in e for e in errors), errors)

    def test_existing_output_and_duplicate_subject_rows_warn(self):
        (self.out / "MDB_0000").mkdir(parents=True)
        (self.out / "MDB_0000" / "leftover").write_text("x")
        _, warnings = self.run_pf(rows=self.rows * 2)
        self.assertTrue(any("overwritten" in w for w in warnings), warnings)
        self.assertTrue(any("2 registry rows" in w for w in warnings), warnings)


# ---------------------------------------------------------------- pipeline CLI

class PipelineCliTests(TempDirCase):
    """braTS_preprocess.py run as the GUI runs it (a subprocess), on tiny volumes with HD-BET skipped/unavailable."""

    def setUp(self):
        super().setUp()
        self.raw = build_export(self.tmp / "raw")
        self.out = self.tmp / "out"

    def run_cli(self, registry_text, *extra, env=None):
        reg = self.registry(registry_text)
        return subprocess.run(
            [sys.executable, str(PIPELINE), "--raw-root", str(self.raw), "--out-root", str(self.out),
             "--registry", str(reg), *extra],
            capture_output=True, text=True, cwd=HERE, env=env, timeout=300)

    def test_no_matching_subject_exits_1_with_message(self):
        r = self.run_cli(HEADER + "\n" + GOOD_ROW + "\n", "--subject", "MDB_9999", "--skip-hdbet")
        self.assertEqual(r.returncode, 1)
        self.assertIn("no registry rows match", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_empty_registry_exits_1(self):
        r = self.run_cli(HEADER + "\n", "--skip-hdbet")
        self.assertEqual(r.returncode, 1)
        self.assertIn("no rows yet", r.stderr)

    def test_malformed_registry_exits_2_without_traceback(self):
        for text in ("", "subject_id, patient_id\nMDB_0000, MB001\n",
                     HEADER + "\nMDB_0000, MB001, 1-Medulloblastoma, 2.10.2010, SER_0004, , SER_0005, SER_0006\n"):
            with self.subTest(text=text[:30]):
                r = self.run_cli(text, "--skip-hdbet")
                self.assertEqual(r.returncode, 2, r.stderr)
                self.assertIn("could not load registry", r.stderr)
                self.assertNotIn("Traceback", r.stderr)

    def test_one_failed_job_does_not_stop_the_next(self):
        bad = "MDB_0000, MB001, 1-Medulloblastoma, 2.10.2010, SER_0099, SER_0003, SER_0005, SER_0006"
        good = "MDB_0001, MB001, 1-Medulloblastoma, 2.10.2010, SER_0004, SER_0003, SER_0005, SER_0006"
        r = self.run_cli(HEADER + f"\n{bad}\n{good}\n", "--skip-hdbet")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("FAILED MDB_0000", r.stderr)
        self.assertIn("1 of 2 job(s) FAILED", r.stderr)
        self.assertTrue((self.out / "MDB_0001" / "02_resampled_1mm" / "t1n.nii.gz").exists())

    def test_results_go_to_per_subject_folder(self):
        r = self.run_cli(HEADER + "\n" + GOOD_ROW + "\n", "--skip-hdbet")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for stage in ("00_raw_nifti", "01_registered", "02_resampled_1mm"):
            self.assertTrue((self.out / "MDB_0000" / stage / "t1c.nii.gz").exists(), stage)

    def test_hdbet_unavailable_fails_cleanly_and_removes_temp_dir(self):
        empty_bin = self.tmp / "emptybin"
        empty_bin.mkdir()
        r = self.run_cli(HEADER + "\n" + GOOD_ROW + "\n", env={"PATH": str(empty_bin)})
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("FAILED MDB_0000", r.stderr)
        self.assertFalse((self.out / "MDB_0000" / "03_final" / "_hdbet_tmp").exists())


# ---------------------------------------------------------------- Tk

def _display_ok():
    try:
        r = tk.Tk()
        r.destroy()
        return True
    except tk.TclError:
        return False


@unittest.skipUnless(_display_ok(), "no display available for Tk")
class GuiTests(TempDirCase):
    def setUp(self):
        super().setUp()
        self.raw = build_export(self.tmp / "raw")
        self.out = self.tmp / "out"
        self.reg = self.registry(HEADER + "\n" + GOOD_ROW + "\n")
        self.errors, self.infos, self.asked = [], [], []
        for name, sink in (("showerror", self.errors), ("showinfo", self.infos)):
            p = mock.patch.object(g.messagebox, name, lambda *a, _s=sink, **k: _s.append(a))
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(g.messagebox, "askyesno", lambda *a, **k: self.asked.append(a) or True)
        p.start()
        self.addCleanup(p.stop)
        self.root = tk.Tk()
        self.app = g.App(self.root)
        self.app.registry_path = self.reg
        self.addCleanup(self._destroy)

    def _destroy(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def pump(self, cond, timeout=30.0):
        end = time.time() + timeout
        while time.time() < end:
            self.root.update()
            if cond():
                return True
            time.sleep(0.02)
        self.root.update()
        return cond()

    def log_text(self):
        return self.app.log.get("1.0", "end")

    def scan(self):
        self.app.raw_var.set(str(self.raw))
        self.app.scan()
        self.assertTrue(self.pump(lambda: str(self.app.scan_btn.cget("state")) == "normal" and self.app.studies))

    def start_run(self, script_text, out=None):
        fake = self.tmp / "fake_pipeline.py"
        fake.write_text(script_text)
        p = mock.patch.object(g, "PREPROCESS_SCRIPT", fake)
        p.start()
        self.addCleanup(p.stop)
        self.app.refresh_registry()
        self.app.raw_root_var.set(str(self.raw))
        self.app.out_var.set(str(out or self.out))
        self.app.skip_hdbet.set(True)
        self.app.reg_tree.selection_set("0")
        self.app.preprocess()

    LONG_RUNNING = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "open(sys.argv[sys.argv.index('--out-root') + 1] + '/child.pid', 'w').write(str(child.pid))\n"
        "print('started', flush=True)\n"
        "time.sleep(120)\n"
    )

    def child_pid(self):
        return int((self.out / "child.pid").read_text())

    # -- scan
    def test_scan_populates_studies_and_suggestions(self):
        self.app.registry_path = self.tmp / "none.csv"
        self.scan()
        self.assertEqual(self.app.raw_root_var.get(), str(self.raw.resolve()))
        self.assertEqual({m: self.app.mod_vars[m].get().split()[0] for m in g.MODALITIES},
                         {"t1n": "SER_0004", "t1c": "SER_0003", "t2w": "SER_0005", "t2f": "SER_0006"})
        self.assertEqual(self.app.subject_var.get(), "MDB_0000")

    def test_scan_survives_one_study_failing_and_an_unreadable_registry(self):
        build_export(self.raw, patients=(("2-Medulloblastoma", "MB002", "5.5.2011"),))
        real = g.scan_study

        def flaky(series_root, progress=None):
            if "2-Medulloblastoma" in str(series_root):
                raise RuntimeError("boom")
            return real(series_root, progress)

        self.app.registry_path = self.registry("")  # empty file: select_modalities.load_registered_keys chokes on it
        with mock.patch.object(g, "scan_study", flaky):
            self.scan()
        states = {s["local_dir"]: s.get("scan_error", "ok") for s in self.app.studies}
        self.assertEqual(states["1-Medulloblastoma"], "ok")
        self.assertIn("boom", states["2-Medulloblastoma"])
        self.assertEqual(self.errors, [])
        # selecting the failed study must not raise, and says why
        self.app.study_tree.selection_set("1")
        self.root.update()
        self.assertIn("scan failed", self.app.study_label.cget("text"))

    def test_scan_of_non_folder_is_reported(self):
        self.app.raw_var.set(str(self.tmp / "missing"))
        self.app.scan()
        self.assertEqual(len(self.errors), 1)

    # -- registry
    def test_unreadable_registry_blocks_add_and_preprocess_with_a_message(self):
        self.scan()
        self.app.registry_path = self.registry("subject_id, patient_id\n")
        self.app.on_study_select()  # must not raise
        self.assertEqual(self.app.subject_var.get(), "")
        self.app.add_to_registry()
        self.assertEqual(len(self.errors), 1)
        self.assertIn("registry can't be used", self.errors[0][1])
        self.app.reg_tree.insert("", "end", iid="x", values=("MDB_0000",) + ("",) * 8)
        self.app.reg_tree.selection_set("x")
        self.app.skip_hdbet.set(True)
        self.app.preprocess()
        self.assertEqual(len(self.errors), 2)
        self.assertFalse(self.app.running)

    def test_add_refuses_series_without_data(self):
        shutil.rmtree(study_root(self.raw) / "SER_0004_T1 MPRAGE")
        (study_root(self.raw) / "SER_0004_T1 MPRAGE" / "OBJ_0001").mkdir(parents=True)
        self.app.registry_path = self.registry(HEADER + "\n")
        self.scan()
        label = next(l for l in self.app.label_to_ser if l.startswith("SER_0004"))
        self.app.mod_vars["t1n"].set(label)
        self.app.add_to_registry()
        self.assertEqual(len(self.errors), 1)
        self.assertIn("no readable image data", self.errors[0][1])
        self.assertEqual(g.read_registry_rows(self.app.registry_path), [])

    def test_add_happy_path_updates_registry_and_next_id(self):
        self.app.registry_path = self.registry(HEADER + "\n")
        self.scan()
        self.app.add_to_registry()
        self.assertEqual(self.errors, [])
        self.assertEqual([r["subject_id"] for r in g.read_registry_rows(self.app.registry_path)], ["MDB_0000"])
        self.assertEqual(self.app.subject_var.get(), "MDB_0001")

    # -- Tk callback errors
    def test_exception_in_a_callback_is_logged_and_shown(self):
        def boom():
            raise ValueError("callback exploded")

        self.root.after(0, boom)
        self.assertTrue(self.pump(lambda: bool(self.errors)))
        self.assertIn("callback exploded", self.errors[0][1])
        self.assertIn("Unexpected error", self.log_text())
        self.assertIn("ValueError", self.log_text())

    # -- preprocess
    def test_preprocess_blocked_by_preflight(self):
        self.start_run("print('should never run')", out=self.raw / "results")  # inside the raw folder
        self.assertTrue(any("inside the raw folder" in e[1] for e in self.errors), self.errors)
        self.assertFalse(self.app.running)
        self.assertFalse((self.raw / "results").exists())

    def test_preprocess_blocked_when_registry_row_points_nowhere(self):
        shutil.rmtree(self.raw / "1-Medulloblastoma")
        self.start_run("print('should never run')")
        self.assertTrue(any("patient folder not found" in e[1] for e in self.errors), self.errors)
        self.assertFalse(self.app.running)

    def test_failed_run_is_reported_and_buttons_recover(self):
        self.start_run("import sys; print('boom on stderr', file=sys.stderr); sys.exit(3)")
        self.assertTrue(self.pump(lambda: not self.app.running))
        self.assertEqual(self.app.reg_tree.set("0", "run_status"), "FAILED (3)")
        self.assertIn("boom on stderr", self.log_text())
        self.assertEqual(str(self.app.run_btn.cget("state")), "normal")
        self.assertEqual(str(self.app.cancel_btn.cget("state")), "disabled")

    def test_successful_run_status_and_output_dir_created(self):
        self.start_run("print('all good')")
        self.assertTrue(self.pump(lambda: not self.app.running))
        self.assertEqual(self.app.reg_tree.set("0", "run_status"), "done")
        self.assertTrue(self.out.is_dir())

    def test_undecodable_output_does_not_kill_the_reader(self):
        self.start_run("import sys; sys.stdout.buffer.write(b'bad \\xff\\xfe bytes\\n'); sys.stdout.flush()")
        self.assertTrue(self.pump(lambda: not self.app.running))
        self.assertEqual(self.app.reg_tree.set("0", "run_status"), "done")

    def test_cancel_kills_the_whole_process_group(self):
        self.start_run(self.LONG_RUNNING)
        self.assertTrue(self.pump(lambda: "started" in self.log_text()))
        pid = self.child_pid()
        self.assertTrue(alive(pid))
        self.app.cancel()
        self.assertTrue(self.pump(lambda: not self.app.running))
        self.assertEqual(self.app.reg_tree.set("0", "run_status"), "cancelled")
        self.assertTrue(self.pump(lambda: not alive(pid), timeout=15), "grandchild process survived Cancel")

    def test_closing_the_window_mid_run_stops_the_pipeline(self):
        self.start_run(self.LONG_RUNNING)
        self.assertTrue(self.pump(lambda: "started" in self.log_text()))
        pid = self.child_pid()
        self.app.on_close()
        self.assertTrue(any("still running" in a[1] for a in self.asked), self.asked)
        self.assertTrue(self._wait_dead(pid), "pipeline process survived closing the window")
        with self.assertRaises(tk.TclError):
            self.root.winfo_exists()

    def _wait_dead(self, pid, timeout=15):
        end = time.time() + timeout
        while time.time() < end and alive(pid):
            time.sleep(0.05)
        return not alive(pid)

    def test_declining_close_keeps_running(self):
        self.start_run(self.LONG_RUNNING)
        self.assertTrue(self.pump(lambda: "started" in self.log_text()))
        pid = self.child_pid()
        with mock.patch.object(g.messagebox, "askyesno", lambda *a, **k: False):
            self.app.on_close()
        self.root.update()
        self.assertTrue(self.root.winfo_exists())
        self.assertTrue(alive(pid))
        self.app.cancel()
        self.assertTrue(self.pump(lambda: not self.app.running))


if __name__ == "__main__":
    unittest.main()
