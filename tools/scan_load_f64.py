#!/usr/bin/env python3
"""Sweep: undeclared load() of float64 data in .tl files (tests/, apps/, examples/, exploratory/).

An undeclared load of a float64 .npy/.npz used to be inferred as f64; from v0.5.6 it is f32.
Run from the repo root. Exit 1 = hits (files whose behaviour changes).
"""
import re, sys
from pathlib import Path
import numpy as np

LOAD = re.compile(r'^\s*let\s+(\w+)\s*=\s*load\(\s*"([^"]+)"', re.M)   # no ": Tensor[...]" annotation
hits, skipped, checked = [], [], 0
for d in ("tests", "apps", "examples", "exploratory"):
    for tl in sorted(Path(d).rglob("*.tl")) if Path(d).is_dir() else []:
        for m in LOAD.finditer(tl.read_text(errors="replace")):
            name, rel = m.groups()
            p = Path(rel)
            if p.suffix.lower() not in (".npy", ".npz"):
                continue
            if not p.is_file():
                skipped.append(f"{tl}: {name} <- {rel} (file not present now; generated at test time?)")
                continue
            try:
                arr = np.load(p)
                if p.suffix.lower() == ".npz":
                    arr = arr["data"] if "data" in arr.files else arr[arr.files[0]]
            except Exception as e:
                skipped.append(f"{tl}: {name} <- {rel} ({e})")
                continue
            checked += 1
            if arr.dtype == np.float64:
                hits.append(f"{tl}: {name} <- {rel} is float64")
print(f"checked {checked} undeclared npy/npz loads")
for s in skipped: print("SKIPPED", s)
for h in hits: print("HIT", h)
print("hits:", len(hits))
sys.exit(1 if hits else 0)
