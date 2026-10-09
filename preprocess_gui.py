"""
Tkinter front-end for the medulloblastoma preprocessing flow:

    1. pick a raw export folder and Scan it
    2. review/override the suggested t1n/t1c/t2w/t2f series per study
    3. "Add to registry" appends a row to subjects_registry.csv
    4. pick an output directory and Preprocess the selected registry rows

All the real work is delegated to the existing scripts -- discovery and
header reads to select_modalities.py / inventory_export.py, preprocessing to
braTS_preprocess.py (run as a subprocess, so the GUI itself never needs
SimpleITK). See GUI_README.md for usage.

Run (from the unet_venv conda env):
    python3 preprocess_gui.py
"""
import contextlib
import csv
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    import fcntl  # POSIX only; without it the registry lock is a no-op
except ImportError:  # pragma: no cover
    fcntl = None

from inventory_export import SER_NAME_RE, _series_files, find_series_roots, inspect_series
from select_modalities import (
    REGISTRY_PATH, classify, load_registered_keys, read_series_headers,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PREPROCESS_SCRIPT = SCRIPT_DIR / "braTS_preprocess.py"
DEFAULT_OUT_ROOT = SCRIPT_DIR / "PreProcessed"

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
REGISTRY_COLUMNS = ["subject_id", "patient_id", "local_dir", "study_date", *MODALITIES]
FLAIR_PATTERN = re.compile(r"FLAIR", re.IGNORECASE)
SUBJECT_ID_RE = re.compile(r"^MDB_(\d+)$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")  # subject_id becomes a directory name
# inspect_series() statuses that mean "possibly truncated" (UNVERIFIED/COMPLETE are not flags)
BAD_STATUSES = {"EMPTY", "SHORT", "GAPS", "AMBIGUOUS-TOTAL", "UNREADABLE", "ERROR"}
# statuses where there is no usable image data at all -- never suggested, never allowed in the registry
NO_DATA_STATUSES = {"EMPTY", "UNREADABLE", "ERROR"}
MIN_FREE_GB = 5.0


class RegistryError(Exception):
    """The registry can't be read/written, or a row would corrupt it."""


# --------------------------------------------------------------------------
# Core logic (no Tk) -- importable and testable headless
# --------------------------------------------------------------------------

def discover_studies(selected: Path):
    """(raw_root, [(local_dir, study_date, series_root), ...]) for `selected`.

    `selected` is normally the folder holding the patient folders (raw/): the
    first path component below it is local_dir, the second is study_date --
    the same labels select_modalities.py / braTS_preprocess.py derive, so
    `raw_root` can be handed straight to `braTS_preprocess.py --raw-root`.
    If `selected` itself directly contains SER_* folders it is treated as one
    study whose parent is the raw root."""
    selected = selected.resolve()
    studies = []
    raw_root = selected
    for series_root in find_series_roots(selected):
        rel = series_root.relative_to(selected)
        if not rel.parts:
            raw_root = selected.parent
            studies.append((selected.name, selected.name, series_root))
        else:
            study_date = rel.parts[1] if len(rel.parts) > 1 else rel.parts[0]
            studies.append((rel.parts[0], study_date, series_root))
    return raw_root, studies


def _error_record(ser_id: str, description: str, status: str, detail: str = "") -> dict:
    return {"ser_id": ser_id, "description": description, "modality": "", "plane": "?", "acq": "",
            "contrast": "", "files": 0, "status": status, "gaps": None, "excluded": True, "error": detail}


def scan_study(series_root: Path, progress=None) -> dict:
    """Series list for one study: description, header facts (plane, 2D/3D,
    contrast, ...) and inspect_series()'s truncation status per series.
    Series without pixel data (PR/KO/SR) are dropped, as in
    select_modalities.py. A series that is empty, unreadable, or that raises
    while being read is KEPT and flagged (EMPTY / UNREADABLE / ERROR) rather
    than being dropped or aborting the whole scan."""
    series, patient_id, patient_name = [], "", ""
    ser_dirs = sorted(d for d in series_root.glob("SER_*") if d.is_dir())
    for i, ser_dir in enumerate(ser_dirs):
        if progress:
            progress(i, len(ser_dirs), ser_dir.name)
        match = SER_NAME_RE.match(ser_dir.name)
        if not match:
            continue
        ser_id, description = match.groups()
        try:
            headers = read_series_headers(ser_dir)
            if headers.get("no_pixel_data") and headers.get("modality"):
                continue  # PR/KO/SR and other non-image objects
            if not headers or headers.get("no_pixel_data"):  # nothing in the folder parsed as an image
                status = "EMPTY" if not _series_files(ser_dir) else "UNREADABLE"
                series.append(_error_record(ser_id, description, status))
                continue
            record = inspect_series(ser_dir)
            patient_id = patient_id or headers.get("patient_id", "")
            patient_name = patient_name or headers.get("patient_name", "")
            series.append({
                "ser_id": ser_id,
                "description": description,
                "modality": headers.get("modality", ""),
                "plane": headers.get("plane", "?"),
                "acq": headers.get("acquisition_type", ""),
                "contrast": headers.get("contrast_agent", ""),
                "files": record.get("on_disk", 0),
                "status": record.get("status", "?"),
                "gaps": record.get("instance_gaps"),
                "excluded": classify(description, headers.get("modality", "")) is None,
                "error": "",
            })
        except Exception as e:  # one bad series must not lose the rest of the study
            series.append(_error_record(ser_id, description, "ERROR", f"{type(e).__name__}: {e}"))
    return {"series": series, "patient_id": patient_id, "patient_name": patient_name}


def suggest_modalities(series: list) -> dict:
    """{"t1n": SER_xxxx|None, ...} using select_modalities.py's own filters
    (classify / EXCLUDE_PATTERN) plus the MODALITY_SELECTION.md preferences:
    pre- vs post-contrast comes from the ContrastBolusAgent TAG, never the
    name; 3D beats 2D; axial beats other planes; more slices break ties.
    Series with no usable data are skipped. Only a suggestion -- the user
    confirms in the GUI."""
    def best(candidates):
        if not candidates:
            return None
        return max(candidates, key=lambda s: (s["acq"] == "3D", s["plane"] == "AX", s["files"]))["ser_id"]

    usable = [s for s in series if s["status"] not in NO_DATA_STATUSES]
    t1 = [s for s in usable if classify(s["description"], s["modality"]) == "t1"]
    t2 = [s for s in usable if classify(s["description"], s["modality"]) == "t2f"]
    t2w_pool = [s for s in t2 if not FLAIR_PATTERN.search(s["description"])]
    t2f_pool = [s for s in t2 if FLAIR_PATTERN.search(s["description"])]

    def pre_first(pool):
        pre = [s for s in pool if not s["contrast"]]
        return pre or pool

    return {
        "t1n": best([s for s in t1 if not s["contrast"]]),
        "t1c": best([s for s in t1 if s["contrast"]]),
        "t2w": best(pre_first(t2w_pool)),
        "t2f": best(pre_first(t2f_pool)),
    }


def read_registry_rows(path: Path) -> list:
    """Registry rows as dicts (all REGISTRY_COLUMNS present, whitespace
    stripped -- the CSV is hand-edited with ', ' padding). A missing or empty
    file is an empty registry. An unreadable file, or a header that lacks a
    required column, raises RegistryError instead of failing later with a
    KeyError."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return []
            names = [(n or "").strip() for n in reader.fieldnames]
            missing = [c for c in REGISTRY_COLUMNS if c not in names]
            if missing:
                raise RegistryError(f"{path.name}: header is missing column(s): {', '.join(missing)}")
            reader.fieldnames = names
            rows = [{c: (row.get(c) or "").strip() for c in REGISTRY_COLUMNS} for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        raise RegistryError(f"cannot read {path}: {e}") from e
    return [r for r in rows if any(r.values())]


def next_subject_id(rows: list) -> str:
    numbers = [int(m.group(1)) for r in rows if (m := SUBJECT_ID_RE.match(r.get("subject_id", "")))]
    return f"MDB_{(max(numbers) + 1 if numbers else 0):04d}"


def _csv_field(value: str) -> str:
    return '"' + value.replace('"', '""') + '"' if re.search(r'[,"\n]', value) else value


def validate_new_row(row: dict, rows: list) -> list:
    """Hard errors that must stop the add (mirrors braTS_preprocess.load_registry's
    checks so a row we write is never one the pipeline refuses to load)."""
    errors = []
    for col in REGISTRY_COLUMNS:
        if not row.get(col, "").strip():
            errors.append(f"'{col}' is empty")
    for col in ("subject_id", "patient_id"):
        if re.search(r"[,\"\n]", row.get(col, "")):
            errors.append(f"'{col}' must not contain commas or quotes")
    sid = row.get("subject_id", "")
    if sid and (not SAFE_ID_RE.match(sid) or sid in (".", "..")):
        errors.append("'subject_id' may only contain letters, digits, '_', '-' and '.' (it becomes a folder name)")
    chosen = [row[m] for m in MODALITIES if row.get(m)]
    if len(set(chosen)) != len(chosen):
        errors.append("the same series is selected for more than one modality")
    for r in rows:
        if r["subject_id"] == row["subject_id"] and r["patient_id"] != row["patient_id"]:
            errors.append(f"{row['subject_id']} already belongs to patient_id {r['patient_id']}")
            break
    key = (row["local_dir"], row["study_date"], row["subject_id"])
    if any((r["local_dir"], r["study_date"], r["subject_id"]) == key for r in rows):
        errors.append("that exact (local_dir, study_date, subject_id) row already exists")
    return errors


@contextlib.contextmanager
def registry_lock(path: Path):
    """Exclusive advisory lock so two GUI instances (or a GUI and a script)
    can't both read-modify-write the registry and lose a row."""
    lock_path = path.with_name(path.name + ".lock")
    f = open(lock_path, "w")
    try:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        f.close()  # closing releases the lock


def add_registry_row(path: Path, row: dict):
    """Validate against the registry's *current* contents and append `row`,
    all under registry_lock(). Raises RegistryError (with every problem found)
    and never leaves a half-written file: the new content goes to a temp file
    that is renamed over the registry (after a .bak copy), in the registry's
    existing ', '-padded style. A field that needs CSV quoting is written
    without the preceding space (csv only recognises a quote at the very
    start of a field)."""
    path = Path(path)
    try:
        with registry_lock(path):
            rows = read_registry_rows(path)
            errors = validate_new_row(row, rows)
            if errors:
                raise RegistryError("\n".join(errors))

            header = ", ".join(REGISTRY_COLUMNS)
            existing = path.read_text() if path.exists() else ""
            if not existing.strip():
                existing = header + "\n"
            else:
                names = [n.strip() for n in existing.splitlines()[0].split(",")]
                if names != REGISTRY_COLUMNS:
                    raise RegistryError(f"{path.name}: columns must be exactly '{header}' (in that order) "
                                        f"to append safely; found '{', '.join(names)}'")
                if not existing.endswith("\n"):
                    existing += "\n"

            line = ""
            for i, col in enumerate(REGISTRY_COLUMNS):
                field = _csv_field(row[col])
                if i:
                    line += "," if field.startswith('"') else ", "
                line += field

            if path.exists():
                shutil.copy2(path, path.with_name(path.name + ".bak"))
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(existing + line + "\n")
            os.replace(tmp, path)
    except OSError as e:
        raise RegistryError(f"cannot write {path}: {e}") from e


def find_study_root(raw_root: Path, local_dir: str, study_date: str) -> Path:
    """The folder holding SER_* dirs for (local_dir, study_date) under raw_root
    -- the same lookup braTS_preprocess.series_root_for() does, without
    importing SimpleITK."""
    patient_dir = Path(raw_root) / local_dir
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"patient folder not found: {patient_dir}")
    for series_root in find_series_roots(patient_dir):
        rel = series_root.relative_to(patient_dir)
        if (rel.parts[0] if rel.parts else patient_dir.name) == study_date:
            return series_root
    raise FileNotFoundError(f"no SER_* folders for study '{study_date}' under {patient_dir}")


def _nearest_existing(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def preflight(rows: list, raw_root, out_root, min_free_gb: float = MIN_FREE_GB):
    """Checks to run BEFORE launching the pipeline, so a bad setup fails with a
    readable message instead of a traceback minutes in. `rows` are the registry
    rows that will run. Returns (errors, warnings): errors block the run,
    warnings need the user's confirmation."""
    errors, warnings = [], []
    raw_root, out_root = Path(raw_root), Path(out_root).expanduser()

    if not PREPROCESS_SCRIPT.is_file():
        errors.append(f"pipeline script not found: {PREPROCESS_SCRIPT}")
    raw_ok = raw_root.is_dir()
    if not raw_ok:
        errors.append(f"raw root is not a folder: {raw_root}")

    out_res = out_root.resolve()
    if raw_ok:
        raw_res = raw_root.resolve()
        if out_res == raw_res or raw_res in out_res.parents:
            errors.append(f"the output directory must not be inside the raw folder ({raw_root})")
    anchor = _nearest_existing(out_res)
    if not anchor.is_dir():
        errors.append(f"output path {anchor} exists but is not a directory")
    elif not os.access(anchor, os.W_OK | os.X_OK):
        errors.append(f"no permission to write under {anchor}")
    else:
        try:
            free = shutil.disk_usage(anchor).free
            if free < min_free_gb * 2 ** 30:
                warnings.append(f"only {free / 2 ** 30:.1f} GB free under {anchor} "
                                f"(each subject can need ~1 GB or more)")
        except OSError as e:
            warnings.append(f"could not check free disk space: {e}")

    seen = {}
    for r in rows:
        label = f"{r.get('subject_id', '?')} ({r.get('local_dir', '?')}/{r.get('study_date', '?')})"
        empty = [c for c in REGISTRY_COLUMNS if not r.get(c)]
        if empty:
            errors.append(f"{label}: registry row has empty field(s): {', '.join(empty)}")
            continue
        seen[r["subject_id"]] = seen.get(r["subject_id"], 0) + 1
        if not raw_ok:
            continue
        try:
            series_root = find_study_root(raw_root, r["local_dir"], r["study_date"])
        except (OSError, ValueError) as e:
            errors.append(f"{label}: {e}")
            continue
        for mod in MODALITIES:
            ser_id = r[mod]
            if (series_root / ser_id).is_dir():
                ser_dir = series_root / ser_id
            else:
                matches = [m for m in series_root.glob(f"{ser_id}_*") if m.is_dir()]
                if len(matches) != 1:
                    errors.append(f"{label}: {mod} = {ser_id}: expected exactly one matching series folder, "
                                  f"found {len(matches)}")
                    continue
                ser_dir = matches[0]
            if not _series_files(ser_dir):
                errors.append(f"{label}: {mod} = {ser_id} has no image files")

    for sid, n in seen.items():
        if n > 1:
            warnings.append(f"{sid} has {n} registry rows; they all run and write into the same {sid}/ folder")
        existing = out_res / sid
        if existing.is_dir() and any(existing.iterdir()):
            warnings.append(f"{existing} already has output; it will be overwritten")
    return errors, warnings


def build_preprocess_cmd(subject_id, raw_root, out_root, registry, skip_hdbet, skip_atlas, device):
    cmd = [sys.executable, "-u", str(PREPROCESS_SCRIPT),
           "--subject", subject_id, "--raw-root", str(raw_root),
           "--out-root", str(out_root), "--registry", str(registry), "--device", device]
    if skip_hdbet:
        cmd.append("--skip-hdbet")
    if skip_atlas:
        cmd.append("--skip-atlas")
    return cmd


def terminate_process_group(proc, grace: float = 5.0):
    """SIGTERM the child's whole process group (python + hd-bet), SIGKILL if it
    is still alive after `grace` seconds. Safe to call on an exited process."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        self.pack(fill="both", expand=True)
        master.title("Medulloblastoma preprocessing")
        master.geometry("1250x900")
        master.protocol("WM_DELETE_WINDOW", self.on_close)
        # Tk swallows exceptions in callbacks (prints to the terminal only);
        # surface them in the log and a dialog instead
        master.report_callback_exception = self._report_exception

        self.q = queue.Queue()
        self.studies = []          # [{local_dir, study_date, series_root, series, patient_id, ...}]
        self.current = None        # study dict currently shown
        self.label_to_ser = {}     # combobox label -> SER id (current study)
        self.proc = None
        self.running = False
        self.cancel_requested = False
        self.registry_path = REGISTRY_PATH

        self.raw_var = tk.StringVar()
        self.raw_root_var = tk.StringVar()
        self.out_var = tk.StringVar(value=str(DEFAULT_OUT_ROOT))
        self.subject_var = tk.StringVar()
        self.patient_var = tk.StringVar()
        self.mod_vars = {m: tk.StringVar() for m in MODALITIES}
        self.skip_hdbet = tk.BooleanVar(value=False)
        self.skip_atlas = tk.BooleanVar(value=False)
        self.device_var = tk.StringVar(value="cpu")

        self._poll_id = None
        self.bind("<Destroy>", self._on_destroy)
        self._build()
        self.refresh_registry()
        self._poll_id = self.after(100, self._poll)

    def _on_destroy(self, event):
        if event.widget is self and self._poll_id:
            with contextlib.suppress(tk.TclError):
                self.after_cancel(self._poll_id)

    # ---- layout ----
    def _build(self):
        top = ttk.LabelFrame(self, text="1. Raw export folder (the folder that contains the patient folders, e.g. raw/)")
        top.pack(fill="x")
        ttk.Entry(top, textvariable=self.raw_var).pack(side="left", fill="x", expand=True, padx=4, pady=4)
        ttk.Button(top, text="Browse…", command=self.browse_raw).pack(side="left", padx=2)
        self.scan_btn = ttk.Button(top, text="Scan", command=self.scan)
        self.scan_btn.pack(side="left", padx=4)
        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", pady=(2, 4))

        mid = ttk.PanedWindow(self, orient="horizontal")
        mid.pack(fill="both", expand=True)

        left = ttk.LabelFrame(mid, text="Studies found")
        self.study_tree = ttk.Treeview(left, columns=("local_dir", "study_date", "patient_id", "state"),
                                       show="headings", height=8, selectmode="browse")
        for col, w in (("local_dir", 130), ("study_date", 90), ("patient_id", 90), ("state", 90)):
            self.study_tree.heading(col, text=col)
            self.study_tree.column(col, width=w)
        self.study_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.study_tree.bind("<<TreeviewSelect>>", self.on_study_select)
        mid.add(left, weight=1)

        right = ttk.LabelFrame(mid, text="2. Series in selected study (grey = excluded type, red = possible truncation / unreadable)")
        cols = ("ser", "description", "plane", "acq", "contrast", "files", "status")
        self.series_tree = ttk.Treeview(right, columns=cols, show="headings", height=8, selectmode="none")
        for col, w in zip(cols, (80, 190, 50, 50, 110, 55, 110)):
            self.series_tree.heading(col, text=col)
            self.series_tree.column(col, width=w)
        self.series_tree.tag_configure("excluded", foreground="#999999")
        self.series_tree.tag_configure("bad", foreground="#c00000")
        self.series_tree.pack(fill="both", expand=True, padx=4, pady=4)
        mid.add(right, weight=3)

        sel = ttk.LabelFrame(self, text="3. Modalities and registry entry")
        sel.pack(fill="x", pady=4)
        self.combos = {}
        for i, mod in enumerate(MODALITIES):
            ttk.Label(sel, text=mod).grid(row=0, column=i * 2, sticky="e", padx=(8, 2), pady=4)
            cb = ttk.Combobox(sel, textvariable=self.mod_vars[mod], state="readonly", width=34)
            cb.grid(row=0, column=i * 2 + 1, padx=(0, 6))
            self.combos[mod] = cb
        ttk.Label(sel, text="subject_id").grid(row=1, column=0, sticky="e", padx=(8, 2))
        ttk.Entry(sel, textvariable=self.subject_var, width=14).grid(row=1, column=1, sticky="w")
        ttk.Label(sel, text="patient_id").grid(row=1, column=2, sticky="e", padx=(8, 2))
        ttk.Entry(sel, textvariable=self.patient_var, width=20).grid(row=1, column=3, sticky="w")
        self.study_label = ttk.Label(sel, text="")
        self.study_label.grid(row=1, column=4, columnspan=3, sticky="w", padx=8)
        self.add_btn = ttk.Button(sel, text="Add to registry", command=self.add_to_registry)
        self.add_btn.grid(row=1, column=7, sticky="e", padx=8, pady=4)

        reg = ttk.LabelFrame(self, text=f"Registry ({self.registry_path.name}) -- select the subjects to preprocess")
        reg.pack(fill="both", expand=True)
        rcols = (*REGISTRY_COLUMNS, "run_status")
        self.reg_tree = ttk.Treeview(reg, columns=rcols, show="headings", height=5, selectmode="extended")
        for col in rcols:
            self.reg_tree.heading(col, text=col)
            self.reg_tree.column(col, width=95)
        self.reg_tree.pack(fill="both", expand=True, side="left", padx=4, pady=4)
        side = ttk.Frame(reg)
        side.pack(side="left", padx=4)
        ttk.Button(side, text="Select all", command=lambda: self.reg_tree.selection_set(self.reg_tree.get_children())).pack(fill="x", pady=2)
        ttk.Button(side, text="Reload", command=self.refresh_registry).pack(fill="x", pady=2)

        run = ttk.LabelFrame(self, text="4. Preprocess")
        run.pack(fill="x", pady=4)
        ttk.Label(run, text="Raw root").grid(row=0, column=0, sticky="e", padx=4)
        ttk.Entry(run, textvariable=self.raw_root_var, width=60).grid(row=0, column=1, sticky="we", pady=2)
        ttk.Button(run, text="Browse…", command=lambda: self._browse_into(self.raw_root_var)).grid(row=0, column=2, padx=4)
        ttk.Label(run, text="Output dir").grid(row=1, column=0, sticky="e", padx=4)
        ttk.Entry(run, textvariable=self.out_var, width=60).grid(row=1, column=1, sticky="we", pady=2)
        ttk.Button(run, text="Browse…", command=lambda: self._browse_into(self.out_var)).grid(row=1, column=2, padx=4)
        opts = ttk.Frame(run)
        opts.grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(opts, text="Skip HD-BET (stop after resample)", variable=self.skip_hdbet).pack(side="left", padx=4)
        ttk.Checkbutton(opts, text="Skip atlas registration", variable=self.skip_atlas).pack(side="left", padx=4)
        ttk.Label(opts, text="Device").pack(side="left", padx=(12, 2))
        ttk.Combobox(opts, textvariable=self.device_var, values=["cpu", "cuda", "mps"], state="readonly", width=6).pack(side="left")
        self.run_btn = ttk.Button(opts, text="Preprocess selected", command=self.preprocess)
        self.run_btn.pack(side="left", padx=(20, 4))
        self.cancel_btn = ttk.Button(opts, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left")
        run.columnconfigure(1, weight=1)

        logf = ttk.LabelFrame(self, text="Log")
        logf.pack(fill="both", expand=True)
        self.log = tk.Text(logf, height=10, wrap="none", state="disabled")
        scroll = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)

    # ---- helpers ----
    def log_line(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip("\n") + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _report_exception(self, exc, val, tb):
        self.log_line("Unexpected error:\n" + "".join(traceback.format_exception(exc, val, tb)))
        messagebox.showerror("Unexpected error", f"{exc.__name__}: {val}\n\nDetails are in the log.")

    def _browse_into(self, var):
        d = filedialog.askdirectory(initialdir=var.get() or str(SCRIPT_DIR))
        if d:
            var.set(d)

    def browse_raw(self):
        start = self.raw_var.get() or str(SCRIPT_DIR / "raw")
        d = filedialog.askdirectory(initialdir=start if Path(start).is_dir() else str(SCRIPT_DIR))
        if d:
            self.raw_var.set(d)

    # ---- scanning ----
    def scan(self):
        folder = Path(self.raw_var.get().strip())
        if not folder.is_dir():
            messagebox.showerror("Scan", f"Not a folder:\n{folder}")
            return
        self.scan_btn.configure(state="disabled")
        self.study_tree.delete(*self.study_tree.get_children())
        self.series_tree.delete(*self.series_tree.get_children())
        self.studies, self.current = [], None
        self.log_line(f"Scanning {folder} ...")
        threading.Thread(target=self._scan_worker, args=(folder,), daemon=True).start()

    def _scan_worker(self, folder):
        try:
            raw_root, found = discover_studies(folder)
            if not found:
                self.q.put(("scan_done", raw_root, []))
                return
            try:
                registered = load_registered_keys(self.registry_path)
            except Exception as e:  # unreadable registry must not block scanning
                registered = set()
                self.q.put(("log", f"  warning: could not read registry to mark known studies: {e}"))
            studies = []
            for n, (local_dir, study_date, series_root) in enumerate(found):
                self.q.put(("progress", n, len(found)))
                self.q.put(("log", f"  [{n + 1}/{len(found)}] {local_dir}/{study_date}"))
                try:
                    info = scan_study(series_root)
                    for s in info["series"]:
                        if s.get("error"):
                            self.q.put(("log", f"    {s['ser_id']} {s['description']}: {s['status']} -- {s['error']}"))
                except Exception as e:  # this study failed; keep scanning the others
                    info = {"series": [], "patient_id": "", "patient_name": "",
                            "scan_error": f"{type(e).__name__}: {e}"}
                    self.q.put(("log", f"    scan failed: {info['scan_error']}"))
                info.update(local_dir=local_dir, study_date=study_date, series_root=series_root,
                            registered=(local_dir, study_date) in registered)
                studies.append(info)
            self.q.put(("scan_done", raw_root, studies))
        except Exception as e:  # surface in the GUI instead of dying silently in the thread
            self.q.put(("scan_error", f"{type(e).__name__}: {e}"))

    def _on_scan_done(self, raw_root, studies):
        self.scan_btn.configure(state="normal")
        self.progress.configure(value=0)
        self.studies = studies
        self.raw_root_var.set(str(raw_root))
        if not studies:
            self.log_line("No SER_* folders found under that folder.")
            return
        for i, s in enumerate(studies):
            state = "SCAN FAILED" if s.get("scan_error") else ("registered" if s["registered"] else "new")
            self.study_tree.insert("", "end", iid=str(i),
                                   values=(s["local_dir"], s["study_date"], s["patient_id"], state))
        flagged = sum(1 for s in studies for x in s["series"] if x["status"] in BAD_STATUSES)
        self.log_line(f"Found {len(studies)} study(ies); raw root = {raw_root}")
        self.log_line(f"{flagged} series flagged for possible truncation / unreadable (red). Note: without a DICOMDIR "
                      f"this check only sees InstanceNumber gaps / declared slice counts -- 'UNVERIFIED' is not a guarantee.")
        first_new = next((i for i, s in enumerate(studies) if not s["registered"] and not s.get("scan_error")), 0)
        self.study_tree.selection_set(str(first_new))

    # ---- study selection ----
    @staticmethod
    def _label(s):
        contrast = s["contrast"] or "no contrast"
        return f"{s['ser_id']}  {s['description']}  [{s['plane']} {s['acq'] or '?'}, {contrast}, {s['files']} files]"

    def on_study_select(self, _event=None):
        sel = self.study_tree.selection()
        if not sel:
            return
        study = self.studies[int(sel[0])]
        self.current = study
        self.series_tree.delete(*self.series_tree.get_children())
        for s in study["series"]:
            tags = ("bad",) if s["status"] in BAD_STATUSES else (("excluded",) if s["excluded"] else ())
            self.series_tree.insert("", "end", tags=tags, values=(
                s["ser_id"], s["description"], s["plane"], s["acq"], s["contrast"] or "none", s["files"], s["status"]))

        labels = [self._label(s) for s in study["series"]]
        self.label_to_ser = {self._label(s): s["ser_id"] for s in study["series"]}
        suggestion = suggest_modalities(study["series"])
        ser_to_label = {v: k for k, v in self.label_to_ser.items()}
        for mod in MODALITIES:
            self.combos[mod].configure(values=labels)
            self.mod_vars[mod].set(ser_to_label.get(suggestion[mod], ""))
        missing = [m for m in MODALITIES if not suggestion[m]]
        try:
            self.subject_var.set(next_subject_id(read_registry_rows(self.registry_path)))
            registry_note = ""
        except RegistryError as e:
            self.subject_var.set("")
            self.log_line(f"Registry problem: {e}")
            registry_note = "  -- REGISTRY UNREADABLE (see log)"
        self.patient_var.set(study["patient_id"])
        note = f"{study['local_dir']}/{study['study_date']}"
        if study.get("scan_error"):
            note += f"  -- scan failed: {study['scan_error']}"
        elif missing:
            note += f"  -- needs selection: {', '.join(missing)}"
        if study["registered"]:
            note += "  -- already in registry"
        self.study_label.configure(text=note + registry_note)

    # ---- registry ----
    def refresh_registry(self):
        keep = set(self.reg_tree.selection())
        self.reg_tree.delete(*self.reg_tree.get_children())
        try:
            rows = read_registry_rows(self.registry_path)
        except RegistryError as e:
            self.log_line(f"Registry problem: {e}")
            return
        for i, r in enumerate(rows):
            self.reg_tree.insert("", "end", iid=str(i), values=(*[r.get(c, "") for c in REGISTRY_COLUMNS], ""))
        self.reg_tree.selection_set([i for i in keep if self.reg_tree.exists(i)])

    def add_to_registry(self):
        study = self.current
        if study is None:
            messagebox.showinfo("Add to registry", "Scan a folder and select a study first.")
            return
        row = {
            "subject_id": self.subject_var.get().strip(),
            "patient_id": self.patient_var.get().strip(),
            "local_dir": study["local_dir"],
            "study_date": study["study_date"],
        }
        for mod in MODALITIES:
            row[mod] = self.label_to_ser.get(self.mod_vars[mod].get(), "")
        try:
            rows = read_registry_rows(self.registry_path)
        except RegistryError as e:
            messagebox.showerror("Cannot add", f"The registry can't be used:\n{e}")
            return
        errors = validate_new_row(row, rows)

        by_ser = {s["ser_id"]: s for s in study["series"]}
        for mod in MODALITIES:  # no data at all: never allowed, not just a warning
            s = by_ser.get(row[mod])
            if s and s["status"] in NO_DATA_STATUSES:
                errors.append(f"{mod} = {s['ser_id']} has no readable image data ({s['status']})")
        if errors:
            messagebox.showerror("Cannot add", "\n".join(f"- {e}" for e in errors))
            return

        warnings = []
        for mod in MODALITIES:
            s = by_ser[row[mod]]
            if s["status"] in BAD_STATUSES:
                warnings.append(f"{mod} = {s['ser_id']} is flagged {s['status']} (possibly truncated)")
            if s["excluded"]:
                warnings.append(f"{mod} = {s['ser_id']} '{s['description']}' is an excluded series type")
        if (row["t1c"] and not by_ser[row["t1c"]]["contrast"]):
            warnings.append(f"t1c = {row['t1c']} has no ContrastBolusAgent in its header (looks pre-contrast)")
        if row["t1n"] and by_ser[row["t1n"]]["contrast"]:
            warnings.append(f"t1n = {row['t1n']} has a ContrastBolusAgent in its header (looks post-contrast)")
        if study["registered"]:
            warnings.append("this study is already in the registry")
        if any(r["patient_id"] == row["patient_id"] for r in rows):
            warnings.append(f"patient_id {row['patient_id']} already appears in the registry under another study "
                            f"(output folders are per subject_id, so use a new subject_id unless you mean to share it)")
        if warnings and not messagebox.askyesno("Add anyway?", "\n".join(f"- {w}" for w in warnings) + "\n\nAdd this row anyway?"):
            return

        try:
            add_registry_row(self.registry_path, row)  # re-validates under a lock against the current file
        except RegistryError as e:
            messagebox.showerror("Add to registry", f"Could not add the row:\n{e}")
            self.refresh_registry()
            return
        study["registered"] = True
        sel = self.study_tree.selection()
        if sel:
            self.study_tree.set(sel[0], "state", "registered")
        with contextlib.suppress(RegistryError):
            self.subject_var.set(next_subject_id(read_registry_rows(self.registry_path)))
        self.log_line(f"Added {row['subject_id']} ({row['local_dir']}/{row['study_date']}) to {self.registry_path.name}"
                      f" (backup: {self.registry_path.name}.bak)")
        self.refresh_registry()

    # ---- preprocessing ----
    def preprocess(self):
        sel = self.reg_tree.selection()
        if not sel:
            messagebox.showinfo("Preprocess", "Select one or more registry rows first.")
            return
        if self.running:
            return
        out_text = self.out_var.get().strip()
        if not out_text:
            messagebox.showerror("Preprocess", "Choose an output directory.")
            return
        out_root = Path(out_text).expanduser()
        raw_root = Path(self.raw_root_var.get().strip())
        if not self.skip_hdbet.get() and shutil.which("hd-bet") is None:
            messagebox.showerror("Preprocess", "hd-bet is not on PATH. Run from the unet_venv environment "
                                 "or tick 'Skip HD-BET'.")
            return
        subjects = []
        for iid in sel:  # one run per unique subject_id; braTS_preprocess.py runs all of its rows
            subject = self.reg_tree.set(iid, "subject_id")
            if subject not in subjects:
                subjects.append(subject)
        try:  # re-read: the table may be stale, and the file is what the pipeline will load
            rows = [r for r in read_registry_rows(self.registry_path) if r["subject_id"] in subjects]
        except RegistryError as e:
            messagebox.showerror("Preprocess", f"The registry can't be used:\n{e}")
            return

        errors, warnings = preflight(rows, raw_root, out_root)
        if errors:
            messagebox.showerror("Cannot start", "\n".join(f"- {e}" for e in errors))
            return
        text = f"Preprocess {len(subjects)} subject(s) into\n{out_root}\n\n{', '.join(subjects)}\n"
        if warnings:
            text += "\nWarnings:\n" + "\n".join(f"- {w}" for w in warnings) + "\n"
        if not messagebox.askyesno("Preprocess", text + "\nThis can take a long time. Continue?"):
            return
        try:
            out_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Preprocess", f"Cannot create output directory:\n{e}")
            return

        self.running = True
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.cancel_requested = False
        for iid in self.reg_tree.get_children():
            self.reg_tree.set(iid, "run_status", "queued" if self.reg_tree.set(iid, "subject_id") in subjects else "")
        args = (subjects, raw_root, out_root, self.registry_path,
                self.skip_hdbet.get(), self.skip_atlas.get(), self.device_var.get())
        threading.Thread(target=self._run_worker, args=args, daemon=True).start()

    def _run_worker(self, subjects, raw_root, out_root, registry, skip_hdbet, skip_atlas, device):
        try:
            for subject in subjects:
                if self.cancel_requested:
                    self.q.put(("status", subject, "cancelled"))
                    continue
                self.q.put(("status", subject, "running"))
                cmd = build_preprocess_cmd(subject, raw_root, out_root, registry, skip_hdbet, skip_atlas, device)
                self.q.put(("log", f"\n$ {' '.join(cmd)}"))
                code = -1
                try:
                    # own session so Cancel can kill hd-bet along with the python child;
                    # stderr merged so tracebacks land in the log; undecodable bytes must not kill the reader
                    with subprocess.Popen(cmd, cwd=SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, errors="replace", start_new_session=True) as proc:
                        self.proc = proc
                        if self.cancel_requested:  # Cancel landed between the check above and the launch
                            terminate_process_group(proc)
                        for line in proc.stdout:
                            self.q.put(("log", line))
                        code = proc.wait()
                except Exception as e:
                    self.q.put(("log", f"Failed to run the pipeline: {type(e).__name__}: {e}"))
                if self.cancel_requested:
                    status = "cancelled"
                else:
                    status = "done" if code == 0 else f"FAILED ({code})"
                self.q.put(("status", subject, status))
        except Exception:  # whatever happens, the buttons must come back
            self.q.put(("log", "Internal error in the run thread:\n" + traceback.format_exc()))
        finally:
            self.proc = None
            self.q.put(("run_done", str(out_root)))

    def cancel(self):
        self.cancel_requested = True
        proc = self.proc
        if proc and proc.poll() is None:
            # don't block the UI for the SIGTERM grace period
            threading.Thread(target=terminate_process_group, args=(proc,), daemon=True).start()
        self.log_line("Cancel requested.")

    def on_close(self):
        if self.running:
            if not messagebox.askyesno("Quit", "Preprocessing is still running. Stop it and quit?"):
                return
            self.cancel_requested = True
            terminate_process_group(self.proc, grace=3.0)  # don't leave python/hd-bet running orphaned
        self.master.destroy()

    # ---- queue pump ----
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self.log_line(msg[1])
                elif kind == "progress":
                    self.progress.configure(maximum=msg[2], value=msg[1])
                elif kind == "scan_done":
                    self._on_scan_done(msg[1], msg[2])
                elif kind == "scan_error":
                    self.scan_btn.configure(state="normal")
                    self.log_line(f"Scan failed: {msg[1]}")
                    messagebox.showerror("Scan failed", msg[1])
                elif kind == "status":
                    for iid in self.reg_tree.get_children():
                        if self.reg_tree.set(iid, "subject_id") == msg[1]:
                            self.reg_tree.set(iid, "run_status", msg[2])
                elif kind == "run_done":
                    self.running = False
                    self.run_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    self.log_line(f"\nFinished. Output under: {msg[1]}")
        except queue.Empty:
            pass
        except Exception:  # a bad message must not kill the pump (and with it all UI updates)
            self.log_line("Internal error in the UI update loop:\n" + traceback.format_exc())
        self._poll_id = self.after(100, self._poll)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
