#!/usr/bin/env python3
"""Compare src/ddfp/cpu_fp.py of the earlier public commit (8fdfe62) with the evaluated version in this tree.

CPU only, small random uint8 volumes.  For each shape and seed it reports the DWC violation count of each
version (benchmark_utils.verify_dwc) and whether the two outputs are identical.  Run from the repository root:
    python cmpb/provenance/check_cpu_fp_public_vs_evaluated.py
This is a code regression check, not an experiment of the manuscript.
"""
import importlib.util, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ddfp import cpu_fp as EVAL                      # noqa: E402
from src.utils.benchmark_utils import verify_dwc          # noqa: E402

src = subprocess.run(["git", "-C", str(ROOT), "show", "8fdfe62:src/ddfp/cpu_fp.py"], capture_output=True, text=True, check=True).stdout
tmp = Path(tempfile.mkdtemp()) / "cpu_fp_public.py"; tmp.write_text(src)
spec = importlib.util.spec_from_file_location("cpu_fp_public", tmp); PUB = importlib.util.module_from_spec(spec); spec.loader.exec_module(PUB)

def run(mod, vol):
    lo, hi, l = mod.build_ispan_cpu(vol)
    u = mod.fp_cpu(lo, hi, l)[1:-1, 1:-1, 1:-1]
    v = verify_dwc(vol[:, :, 0], u[:, :, 0]) if vol.shape[2] == 1 else verify_dwc(vol, u)   # depth-1 input = 2-D
    return u, v["n_violations"]

n_diff = 0
for shape in [(4, 4, 4), (8, 8, 8), (12, 10, 1), (16, 9, 1), (8, 8, 5), (5, 8, 8), (5, 6, 7), (6, 5, 4), (3, 8, 5), (7, 3, 4)]:
    for seed in range(3):
        vol = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
        (up, vp), (ue, ve) = run(PUB, vol), run(EVAL, vol)
        same = bool(np.array_equal(up, ue)); n_diff += not same
        print(f"{str(shape):12s} seed {seed}  violations public/evaluated: {vp:3d}/{ve:3d}  identical: {same}")
print(f"{n_diff} of 30 cases differ")
