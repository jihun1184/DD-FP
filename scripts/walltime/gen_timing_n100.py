"""
scripts/walltime/gen_timing_n100.py  (K=16 variant)
------------------------------------------------------
Generate timing_k16.csv: BraTS2021 N=100 subjects wall-clock timing.

Changes from brats_n100_sampling.py:
  v4: seq_cpu uses build_ispan_cpu() + fp_cpu() (pure CPU sequential FP),
      consistent with measure_cpu_brats.py.
  v3: added --K argument (default=16) for K=16 timing.
  v2: GPU timing uses perf_counter (avoids CuPy Event API compatibility issues).

Usage:
    # K=16, timing_k16.csv
    python scripts/walltime/gen_timing_n100.py \
        --brats-dir /path/to/BraTS2021_Training \
        --output timing_k16.csv \
        --n-subjects 100 --random-seed 42 --skip-cpu --K 16

    # K=16 with seq_cpu
    python scripts/walltime/gen_timing_n100.py \
        --brats-dir /path/to/BraTS2021_Training \
        --output timing_k16_withcpu.csv \
        --n-subjects 20 --random-seed 42 --K 16
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _candidate in [_HERE, _HERE.parent, _HERE.parent.parent]:
    if (_candidate / "src" / "ddfp").exists():
        sys.path.insert(0, str(_candidate))
        break

# ── GPU imports ───────────────────────────────────────────────────────────────
try:
    import cupy as cp
    from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu
    _CUPY_OK = True
except Exception as e:
    _CUPY_OK = False
    print(f"[WARN] CuPy/GPU unavailable: {e}")

try:
    import nibabel as nib
    _NIB_OK = True
except ImportError:
    _NIB_OK = False
    print("[WARN] nibabel not found — pip install nibabel")

try:
    from src.utils.benchmark_utils import verify_dwc
    _BENCH_OK = True
except Exception:
    _BENCH_OK = False


# ── Constants ─────────────────────────────────────────────────────────────────
EXPANDED_CELLS = 240 * 240 * 155 * 7   # ≈ 62.5M (BraTS expanded lattice)

FIELDNAMES = [
    "subject_id", "t_cpu_s", "t_gpu1_s", "t_dd_s",
    "speedup_gpu1", "speedup_dd", "r_star", "viols_dd",
    "dwc_guaranteed_dd", "tips_cpu", "tips_gpu1", "tips_dd",
]


def _split_domain_1d(D: int, K: int) -> list[tuple[int, int]]:
    base = D // K
    rem  = D % K
    slices, z = [], 0
    for k in range(K):
        size = base + (1 if k < rem else 0)
        slices.append((z, z + size))
        z += size
    return slices


# ── Boundary helpers (exp_a3 verbatim) ───────────────────────────────────────

def _pin_and_seed_expanded(U_lo, U_hi, seeds, u_assembled,
                            z0_ext, gz_start, gz_end, W2s, H2s):
    n_g = gz_end - gz_start
    l_pad_start = (gz_start - 2 * z0_ext) + 1
    n_l = min(n_g, U_lo.shape[2] - l_pad_start)
    actual = min(n_g, n_l)
    if actual <= 0:
        return
    vals = u_assembled[:W2s, :H2s, gz_start:gz_start + actual]
    U_lo[1:W2s+1, 1:H2s+1, l_pad_start:l_pad_start + actual] = vals
    U_hi[1:W2s+1, 1:H2s+1, l_pad_start:l_pad_start + actual] = vals
    seeds[1:W2s+1, 1:H2s+1, l_pad_start:l_pad_start + actual] = vals


def _assemble_boundary(u_target, u_sub, z_b, z0_ext, D2):
    local_zb = z_b - z0_ext
    ez_0 = 2 * local_zb;  ez_1 = ez_0 - 1
    gz_0 = 2 * z_b;       gz_1 = gz_0 - 1
    if 0 <= ez_0 < u_sub.shape[2] and gz_0 < D2:
        u_target[:, :, gz_0] = u_sub[:, :, ez_0]
    if 0 <= ez_1 < u_sub.shape[2] and gz_1 >= 0:
        u_target[:, :, gz_1] = u_sub[:, :, ez_1]


def run_ibi_v10(vol_u8: np.ndarray, K: int = 4, delta: int = 1,
                max_rounds: int = 8, l_inf: float = 0.0) -> dict:
    """
    IBI DD-FP (K=args.K, delta=1). K is determined by the CLI argument.
    Same implementation as run_ibi_v10 in exp_a3_delta.py.
    Returns: {u_dd, R_star, t_total_s, boundary_z_orig}
    """
    W, H, D = vol_u8.shape
    W2, H2, D2 = 2*W - 1, 2*H - 1, 2*D - 1
    W2s, H2s = W2, H2
    t0 = time.perf_counter()

    K_eff = min(K, D)
    slices_z = [(z0, z1) for z0, z1 in _split_domain_1d(D, K_eff) if z1 > z0]
    boundary_z_orig = [z1 for _, z1 in slices_z[:-1]]

    # Round 0: independent FP per subdomain
    u_assembled = cp.zeros((W2, H2, D2), dtype=cp.float32)
    sub_meta = []

    for k, (z0, z1) in enumerate(slices_z):
        z0_ext = max(0, z0 - delta)
        z1_ext = min(D, z1 + delta)
        sub_vol = vol_u8[:, :, z0_ext:z1_ext]

        U_lo, U_hi, _ = build_ispan_gpu(sub_vol)
        u_sub_pad = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)
        u_sub = u_sub_pad[1:-1, 1:-1, 1:-1]

        inner_loc_z0 = z0 - z0_ext
        inner_loc_z1 = inner_loc_z0 + (z1 - z0)
        ez_s = 2 * inner_loc_z0 if k > 0 else 0
        ez_e = 2 * inner_loc_z1 - 1 if k < len(slices_z) - 1 else u_sub.shape[2]
        gz_s = 2 * z0 if k > 0 else 0
        gz_e = min(gz_s + (ez_e - ez_s), D2)
        actual = gz_e - gz_s
        if actual > 0 and ez_s + actual <= u_sub.shape[2]:
            u_assembled[:, :, gz_s:gz_e] = u_sub[:, :, ez_s:ez_s + actual]
        if k < len(slices_z) - 1:
            _assemble_boundary(u_assembled, u_sub, z1, z0_ext, D2)

        sub_meta.append({"k": k, "z0": z0, "z1": z1,
                         "z0_ext": z0_ext, "z1_ext": z1_ext})

    # IBI rounds
    R_star = 0
    for r in range(1, max_rounds + 1):
        u_prev = u_assembled.copy()
        u_new  = u_assembled.copy()

        for m in sub_meta:
            k = m["k"]; z0 = m["z0"]; z1 = m["z1"]
            z0_ext = m["z0_ext"]; z1_ext = m["z1_ext"]
            sub_vol = vol_u8[:, :, z0_ext:z1_ext]

            U_lo, U_hi, _ = build_ispan_gpu(sub_vol)
            seeds = cp.full(U_lo.shape, cp.nan, dtype=cp.float32)

            if k > 0:
                _pin_and_seed_expanded(U_lo, U_hi, seeds, u_assembled,
                    z0_ext, 2*z0_ext, 2*z0+1, W2s, H2s)
            if k < len(slices_z) - 1:
                _pin_and_seed_expanded(U_lo, U_hi, seeds, u_assembled,
                    z0_ext, 2*z1, 2*z1_ext, W2s, H2s)

            u_sub_pad = front_propagation_gpu(U_lo, U_hi, l_inf,
                                              boundary_seeds=seeds, verbose=False)
            u_sub = u_sub_pad[1:-1, 1:-1, 1:-1]

            inner_loc_z0 = z0 - z0_ext
            inner_loc_z1 = inner_loc_z0 + (z1 - z0)
            ez_s = 2 * inner_loc_z0 if k > 0 else 0
            ez_e = 2 * inner_loc_z1 - 1 if k < len(slices_z) - 1 else u_sub.shape[2]
            gz_s = 2 * z0 if k > 0 else 0
            gz_e = min(gz_s + (ez_e - ez_s), D2)
            actual = gz_e - gz_s
            if actual > 0 and ez_s + actual <= u_sub.shape[2]:
                u_new[:, :, gz_s:gz_e] = u_sub[:, :, ez_s:ez_s + actual]
            if k < len(slices_z) - 1:
                _assemble_boundary(u_new, u_sub, z1, z0_ext, D2)

        max_change = float(cp.abs(u_new - u_prev).max().item())
        u_assembled = u_new
        R_star = r
        if max_change < 0.5:
            break

    cp.cuda.Stream.null.synchronize()
    return {
        "u_dd": cp.asnumpy(u_assembled),
        "boundary_z_orig": boundary_z_orig,
        "R_star": R_star,
        "t_total_s": time.perf_counter() - t0,
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_brats_seg(nii_path: Path) -> np.ndarray:
    img  = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)
    if data.shape[-1] < data.shape[0]:
        data = np.transpose(data, (2, 0, 1))   # (H,W,D)→(D,H,W)
    return (data > 0.5).astype(np.uint8)


def find_seg_file(subject_dir: Path) -> Path | None:
    for pat in ["*_seg.nii.gz", "*_seg.nii", "*seg*.nii.gz"]:
        hits = sorted(subject_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def collect_subjects(brats_dir: Path, n: int, seed: int) -> list[Path]:
    all_dirs = sorted([
        d for d in brats_dir.iterdir()
        if d.is_dir() and d.name.startswith("BraTS")
    ])
    if not all_dirs:
        raise FileNotFoundError(f"No BraTS subjects in {brats_dir}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_dirs), size=min(n, len(all_dirs)), replace=False)
    return [all_dirs[i] for i in sorted(idx)]


# ── Per-subject timing ────────────────────────────────────────────────────────

def _time_gpu1(vol_whd: np.ndarray, n_repeats: int) -> float:
    """GPU single-pass (K=1). Uses perf_counter to avoid CuPy Event API issues."""
    times = []
    for i in range(n_repeats + 1):
        t0 = time.perf_counter()
        U_lo, U_hi, l_inf = build_ispan_gpu(vol_whd)
        _ = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        if i > 0:
            times.append(elapsed)
    return float(np.median(times))


def _time_dd(vol_whd: np.ndarray, n_repeats: int,
             K: int = 16, delta: int = 1) -> tuple[float, int, int]:
    """DD-FP IBI (K=args.K, delta=1) using run_ibi_v10."""
    times, r_stars, viols = [], [], []
    for i in range(n_repeats + 1):
        result = run_ibi_v10(vol_whd, K=K, delta=delta)
        if i > 0:
            times.append(result["t_total_s"])
            r_stars.append(result["R_star"])
            u_dd = result["u_dd"]
            n_viols = 0
            if _BENCH_OK:
                ref_bin = (vol_whd > 0).astype(np.uint8)
                u_norm = u_dd / 255.0 if u_dd.max() > 1.0 else u_dd
                n_viols = verify_dwc(ref_bin, u_norm).get("n_violations", 0)
            viols.append(n_viols)
    return (float(np.median(times)),
            int(np.median(r_stars)),
            int(np.median(viols)))


def process_subject(sid: str, seg_path: Path,
                    n_repeats: int, skip_cpu: bool,
                    K: int = 16) -> dict | None:
    try:
        vol_dhw = load_brats_seg(seg_path)           # (D,H,W)
        vol_whd = vol_dhw.transpose(2, 1, 0).copy()  # (W,H,D) ← GPU convention
    except Exception as e:
        print(f"  [ERROR] {sid}: load — {e}")
        return None

    if vol_dhw.sum() == 0:
        print(f"  [SKIP]  {sid}: empty mask")
        return None

    print(f"  {sid}", end="", flush=True)

    t_cpu = None
    if not skip_cpu:
        try:
            from experiment_DDFP_all import build_ispan_cpu, fp_cpu
            times = []
            for i in range(n_repeats + 1):
                t0 = time.perf_counter()
                U_lo, U_hi, li = build_ispan_cpu(vol_whd)
                _ = fp_cpu(U_lo, U_hi, li)[1:-1, 1:-1, 1:-1]
                elapsed = time.perf_counter() - t0
                if i > 0:
                    times.append(elapsed)
            t_cpu = float(np.median(times))
            print(f"  cpu={t_cpu:.1f}s", end="", flush=True)
        except Exception as e:
            print(f"  cpu=ERR({e})", end="", flush=True)

    # GPU single-pass
    t_gpu1 = None
    if _CUPY_OK:
        try:
            t_gpu1 = _time_gpu1(vol_whd, n_repeats)
            print(f"  gpu1={t_gpu1:.3f}s", end="", flush=True)
        except Exception as e:
            print(f"  gpu1=ERR({e})", end="", flush=True)

    # DD-FP IBI
    t_dd, r_star, viols_dd = None, 2, 0
    if _CUPY_OK:
        try:
            t_dd, r_star, viols_dd = _time_dd(vol_whd, n_repeats, K=K)
            print(f"  dd={t_dd:.2f}s r*={r_star} viols={viols_dd}", end="", flush=True)
        except Exception as e:
            print(f"  dd=ERR({e})", end="", flush=True)

    print()

    dwc_ok = (viols_dd == 0)

    def _tips(t, guaranteed):
        if not t or t == 0:
            return None
        return round(EXPANDED_CELLS / t) if guaranteed else 0

    speedup_gpu1 = round(t_cpu / t_gpu1, 1) if (t_cpu and t_gpu1) else None
    speedup_dd   = round(t_cpu / t_dd,   1) if (t_cpu and t_dd)   else None

    return {
        "subject_id":        sid,
        "t_cpu_s":           round(t_cpu,  2) if t_cpu  else None,
        "t_gpu1_s":          round(t_gpu1, 3) if t_gpu1 else None,
        "t_dd_s":            round(t_dd,   2) if t_dd   else None,
        "speedup_gpu1":      speedup_gpu1,
        "speedup_dd":        speedup_dd,
        "r_star":            r_star,
        "viols_dd":          viols_dd,
        "dwc_guaranteed_dd": dwc_ok,
        "tips_cpu":          _tips(t_cpu,  True),
        "tips_gpu1":         0,
        "tips_dd":           _tips(t_dd, dwc_ok),
    }


# ── Summary stats ─────────────────────────────────────────────────────────────

def append_summary_stats(rows: list[dict]) -> list[dict]:
    numeric = ["t_cpu_s", "t_gpu1_s", "t_dd_s",
               "speedup_gpu1", "speedup_dd",
               "r_star", "viols_dd",
               "tips_cpu", "tips_gpu1", "tips_dd"]
    data = [r for r in rows if not str(r.get("subject_id","")).startswith("__")]

    def _stat(name, fn):
        row = {"subject_id": name}
        for col in FIELDNAMES[1:]:
            if col not in numeric:
                row[col] = ""
                continue
            vals = [float(r[col]) for r in data
                    if r.get(col) not in (None, "", "None")]
            row[col] = round(fn(vals), 3) if vals else ""
        return row

    return rows + [
        _stat("__STAT_MEAN__",   np.mean),
        _stat("__STAT_STD__",    np.std),
        _stat("__STAT_MIN__",    np.min),
        _stat("__STAT_MAX__",    np.max),
        _stat("__STAT_MEDIAN__", np.median),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    if not _NIB_OK:
        sys.exit("nibabel required: pip install nibabel")
    if not _CUPY_OK:
        sys.exit("CuPy required for GPU timing.")

    brats_dir = Path(args.brats_dir)
    if not brats_dir.exists():
        sys.exit(f"BraTS dir not found: {brats_dir}")

    print(f"\n{'='*65}")
    print(f"  gen_timing_n100.py  v3 (K={args.K})")
    print(f"  BraTS  : {brats_dir}")
    print(f"  N      : {args.n_subjects}  seed={args.random_seed}")
    print(f"  K      : {args.K}  delta=1")
    print(f"  Repeats: {args.n_repeats}  (warmup=1 discarded)")
    print(f"  skip_cpu: {args.skip_cpu}")
    print(f"{'='*65}\n")

    subject_dirs = collect_subjects(brats_dir, args.n_subjects, args.random_seed)
    print(f"Sampled {len(subject_dirs)} subjects\n")

    rows: list[dict] = []
    processed = 0

    for subj_dir in subject_dirs:
        sid = subj_dir.name
        seg_file = find_seg_file(subj_dir)
        if seg_file is None:
            print(f"  [SKIP] {sid}: no seg file")
            continue
        row = process_subject(sid, seg_file, args.n_repeats, args.skip_cpu, K=args.K)
        if row is not None:
            rows.append(row)
            processed += 1

    if not rows:
        sys.exit("No subjects processed.")

    rows = append_summary_stats(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    data_rows = [r for r in rows if not str(r["subject_id"]).startswith("__")]
    t_dd_vals  = [float(r["t_dd_s"])  for r in data_rows if r.get("t_dd_s")]
    t_g1_vals  = [float(r["t_gpu1_s"]) for r in data_rows if r.get("t_gpu1_s")]
    dwc_pass   = sum(1 for r in data_rows if r.get("dwc_guaranteed_dd") in (True,"True"))

    print(f"\n{'='*65}")
    print(f"  DONE — {processed} subjects")
    print(f"  Output : {out}")
    if t_dd_vals:
        print(f"  DD-FP  : mean={np.mean(t_dd_vals):.2f}s  "
              f"median={np.median(t_dd_vals):.2f}s")
    if t_g1_vals:
        print(f"  GPU1   : mean={np.mean(t_g1_vals):.3f}s  "
              f"median={np.median(t_g1_vals):.3f}s")
    print(f"  DWC pass: {dwc_pass}/{processed}")
    print(f"\n  → A2 BRATS_REF_ROWS values updated:")
    if t_dd_vals:
        print(f"      t_dd_s  median = {np.median(t_dd_vals):.3f}")
    if t_g1_vals:
        print(f"      t_gpu1_s median = {np.median(t_g1_vals):.3f}")
    print(f"      note = 'BraTS2021 N={processed} median (timing_n100.csv)'")
    print(f"{'='*65}\n")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate timing CSV (K-configurable)")
    p.add_argument("--brats-dir", required=True)
    p.add_argument("--output", default="timing_n100.csv")
    p.add_argument("--n-subjects",  type=int, default=100)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--n-repeats",   type=int, default=5)
    p.add_argument("--skip-cpu", action="store_true",
                   help="Skip sequential CPU timing (~184s/subject)")
    p.add_argument("--K", type=int, default=16,
                   help="DD-FP subdomain count (default=16 for K=16)")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())