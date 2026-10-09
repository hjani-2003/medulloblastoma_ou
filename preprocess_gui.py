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
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from inventory_export import SER_NAME_RE, find_series_roots, inspect_series
from select_modalities import (
    EXCLUDE_PATTERN, REGISTRY_PATH, classify, load_registered_keys, read_series_headers,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PREPROCESS_SCRIPT = SCRIPT_DIR / "braTS_preprocess.py"
DEFAULT_OUT_ROOT = SCRIPT_DIR / "PreProcessed"

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
REGISTRY_COLUMNS = ["subject_id", "patient_id", "local_dir", "study_date", *MODALITIES]
FLAIR_PATTERN = re.compile(r"FLAIR", re.IGNORECASE)
SUBJECT_ID_RE = re.compile(r"^MDB_(\d+)$")
# inspect_series() statuses that mean "possibly truncated" (UNVERIFIED/COMPLETE are not flags)
BAD_STATUSES = {"EMPTY", "SHORT", "GAPS", "AMBIGUOUS-TOTAL", "UNREADABLE"}


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


def scan_study(series_root: Path, progress=None) -> dict:
    """Series list for one study: description, header facts (plane, 2D/3D,
    contrast, ...) and inspect_series()'s truncation status per series.
    Series without pixel data (PR/KO/SR, localizer stubs) are dropped, as in
    select_modalities.py."""
    series, patient_id, patient_name = [], "", ""
    ser_dirs = sorted(d for d in series_root.glob("SER_*") if d.is_dir())
    for i, ser_dir in enumerate(ser_dirs):
        if progress:
            progress(i, len(ser_dirs), ser_dir.name)
        match = SER_NAME_RE.match(ser_dir.name)
        if not match:
            continue
        ser_id, description = match.groups()
        headers = read_series_headers(ser_dir)
        if not headers or headers.get("no_pixel_data"):
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
        })
    return {"series": series, "patient_id": patient_id, "patient_name": patient_name}


def suggest_modalities(series: list) -> dict:
    """{"t1n": SER_xxxx|None, ...} using select_modalities.py's own filters
    (classify / EXCLUDE_PATTERN) plus the MODALITY_SELECTION.md preferences:
    pre- vs post-contrast comes from the ContrastBolusAgent TAG, never the
    name; 3D beats 2D; axial beats other planes; more slices break ties.
    Only a suggestion -- the user confirms in the GUI."""
    def best(candidates):
        if not candidates:
            return None
        return max(candidates, key=lambda s: (s["acq"] == "3D", s["plane"] == "AX", s["files"]))["ser_id"]

    t1 = [s for s in series if classify(s["description"], s["modality"]) == "t1"]
    t2 = [s for s in series if classify(s["description"], s["modality"]) == "t2f"]
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
    """Registry rows as dicts with whitespace stripped (the CSV is
    hand-edited with ', ' padding)."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        reader.fieldnames = [(n or "").strip() for n in reader.fieldnames]
        return [{k: (v or "").strip() for k, v in row.items() if k} for row in reader]


def next_subject_id(rows: list) -> str:
    numbers = [int(m.group(1)) for r in rows if (m := SUBJECT_ID_RE.match(r.get("subject_id", "")))]
    return f"MDB_{(max(numbers) + 1 if numbers else 0):04d}"


def _csv_field(value: str) -> str:
    return '"' + value.replace('"', '""') + '"' if re.search(r'[,"\n]', value) else value


def append_registry_row(path: Path, row: dict):
    """Append one row in the registry's existing ', '-padded style. Writes a
    temp file and renames it over the registry (after a .bak copy), so a crash
    mid-write can't leave a half-written registry. A field that needs CSV
    quoting is written without the preceding space (csv only recognises a
    quote at the very start of a field)."""
    header = ", ".join(REGISTRY_COLUMNS)
    existing = path.read_text() if path.exists() else ""
    if not existing.strip():
        existing = header + "\n"
    elif not existing.endswith("\n"):
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


def validate_new_row(row: dict, rows: list) -> list:
    """Hard errors that must stop the add (mirrors braTS_preprocess.load_registry's
    checks so a row we write is never one the pipeline refuses to load)."""
    errors = []
    for col in REGISTRY_COLUMNS:
        if not row[col].strip():
            errors.append(f"'{col}' is empty")
    for col in ("subject_id", "patient_id"):
        if re.search(r"[,\"\n]", row[col]):
            errors.append(f"'{col}' must not contain commas or quotes")
    chosen = [row[m] for m in MODALITIES if row[m]]
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


def build_preprocess_cmd(subject_id, raw_root, out_root, registry, skip_hdbet, skip_atlas, device):
    cmd = [sys.executable, "-u", str(PREPROCESS_SCRIPT),
           "--subject", subject_id, "--raw-root", str(raw_root),
           "--out-root", str(out_root), "--registry", str(registry), "--device", device]
    if skip_hdbet:
        cmd.append("--skip-hdbet")
    if skip_atlas:
        cmd.append("--skip-atlas")
    return cmd


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        self.pack(fill="both", expand=True)
        master.title("Medulloblastoma preprocessing")
        master.geometry("1250x900")

        self.q = queue.Queue()
        self.studies = []          # [{local_dir, study_date, series_root, series, patient_id, ...}]
        self.current = None        # study dict currently shown
        self.label_to_ser = {}     # combobox label -> SER id (current study)
        self.proc = None
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

        self._build()
        self.refresh_registry()
        self.after(100, self._poll)

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

        right = ttk.LabelFrame(mid, text="2. Series in selected study (grey = excluded type, red = possible truncation)")
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
            registered = load_registered_keys(self.registry_path)
            studies = []
            for n, (local_dir, study_date, series_root) in enumerate(found):
                self.q.put(("progress", n, len(found)))
                self.q.put(("log", f"  [{n + 1}/{len(found)}] {local_dir}/{study_date}"))
                info = scan_study(series_root)
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
            state = "registered" if s["registered"] else "new"
            self.study_tree.insert("", "end", iid=str(i),
                                   values=(s["local_dir"], s["study_date"], s["patient_id"], state))
        flagged = sum(1 for s in studies for x in s["series"] if x["status"] in BAD_STATUSES)
        self.log_line(f"Found {len(studies)} study(ies); raw root = {raw_root}")
        self.log_line(f"{flagged} series flagged for possible truncation (red). Note: without a DICOMDIR this check "
                      f"only sees InstanceNumber gaps / declared slice counts -- 'UNVERIFIED' is not a guarantee.")
        first_new = next((i for i, s in enumerate(studies) if not s["registered"]), 0)
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
        rows = read_registry_rows(self.registry_path)
        self.subject_var.set(next_subject_id(rows))
        self.patient_var.set(study["patient_id"])
        note = f"{study['local_dir']}/{study['study_date']}"
        if missing:
            note += f"  -- needs selection: {', '.join(missing)}"
        if study["registered"]:
            note += "  -- already in registry"
        self.study_label.configure(text=note)

    # ---- registry ----
    def refresh_registry(self):
        keep = set(self.reg_tree.selection())
        self.reg_tree.delete(*self.reg_tree.get_children())
        try:
            rows = read_registry_rows(self.registry_path)
        except Exception as e:
            self.log_line(f"Could not read registry: {e}")
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
        rows = read_registry_rows(self.registry_path)
        errors = validate_new_row(row, rows)
        if errors:
            messagebox.showerror("Cannot add", "\n".join(f"- {e}" for e in errors))
            return

        by_ser = {s["ser_id"]: s for s in study["series"]}
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
            append_registry_row(self.registry_path, row)
        except Exception as e:
            messagebox.showerror("Add to registry", f"Could not write registry:\n{e}")
            return
        study["registered"] = True
        sel = self.study_tree.selection()
        if sel:
            self.study_tree.set(sel[0], "state", "registered")
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
        out_root = Path(self.out_var.get().strip())
        raw_root = Path(self.raw_root_var.get().strip())
        if not self.out_var.get().strip():
            messagebox.showerror("Preprocess", "Choose an output directory.")
            return
        if not raw_root.is_dir():
            messagebox.showerror("Preprocess", f"Raw root is not a folder:\n{raw_root}")
            return
        if not self.skip_hdbet.get() and shutil.which("hd-bet") is None:
            messagebox.showerror("Preprocess", "hd-bet is not on PATH. Run from the unet_venv environment "
                                 "or tick 'Skip HD-BET'.")
            return
        subjects = []
        for iid in sel:  # one run per unique subject_id; braTS_preprocess.py runs all of its rows
            subject = self.reg_tree.set(iid, "subject_id")
            if subject not in subjects:
                subjects.append(subject)
        if not messagebox.askyesno("Preprocess", f"Preprocess {len(subjects)} subject(s) into\n{out_root}\n\n"
                                   f"{', '.join(subjects)}\n\nThis can take a long time."):
            return
        try:
            out_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Preprocess", f"Cannot create output directory:\n{e}")
            return

        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.cancel_requested = False
        for iid in self.reg_tree.get_children():
            self.reg_tree.set(iid, "run_status", "queued" if self.reg_tree.set(iid, "subject_id") in subjects else "")
        args = (subjects, raw_root, out_root, self.registry_path,
                self.skip_hdbet.get(), self.skip_atlas.get(), self.device_var.get())
        threading.Thread(target=self._run_worker, args=args, daemon=True).start()

    def _run_worker(self, subjects, raw_root, out_root, registry, skip_hdbet, skip_atlas, device):
        for subject in subjects:
            if self.cancel_requested:
                self.q.put(("status", subject, "cancelled"))
                continue
            self.q.put(("status", subject, "running"))
            cmd = build_preprocess_cmd(subject, raw_root, out_root, registry, skip_hdbet, skip_atlas, device)
            self.q.put(("log", f"\n$ {' '.join(cmd)}"))
            try:
                # own session so Cancel can kill hd-bet along with the python child
                self.proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, start_new_session=True)
                for line in self.proc.stdout:
                    self.q.put(("log", line))
                code = self.proc.wait()
            except Exception as e:
                self.q.put(("log", f"Failed to launch: {e}"))
                code = -1
            if self.cancel_requested:
                status = "cancelled"
            else:
                status = "done" if code == 0 else f"FAILED ({code})"
            self.q.put(("status", subject, status))
        self.proc = None
        self.q.put(("run_done", str(out_root)))

    def cancel(self):
        self.cancel_requested = True
        proc = self.proc
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
        self.log_line("Cancel requested.")

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
                    self.run_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    self.log_line(f"\nFinished. Output under: {msg[1]}")
        except queue.Empty:
            pass
        self.after(100, self._poll)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
