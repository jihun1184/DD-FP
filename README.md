# DD-FP: Domain-Decomposed Front Propagation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**DD-FP** produces a Digitally Well-Composed (DWC) interpolation field from a 2-/3-D biomedical (binary/gray-scale) datasets via parallel Level-BFS front propagation and Iterative Boundary Injection (IBI). Our main contributions are threefold:
* **Uniqueness of $\delta=1$ and Finite-Round IBI Convergence**
   * We prove that a halo width of $\delta=1$ is the **uniquely optimal choice** (Corollary 3.14)
     * $\delta=0$ directly violates DWC correctness due to propagation path severance.
     * $\delta\ge2$ breaks the inflationary guarantee of the assembled operator, making the convergence bound ($R^*$) image- and partition-dependent rather than structurally assured.
   * At $\delta=1$, the proposed **Iterative Boundary Injection (IBI)** structurally guarantees convergence to the global DWC optimum in **$R^ \le 2$ rounds** for any subdomain count $K$ satisfying the boundary-seed coverage condition (A1).
   * We show this bound is tight ($R^*=2$) at $K=16$ across all 1,251 BraTS 2021 subjects.
* **Scalable GPU Implementation with Boundary-Seed Correction**
   * We replace the sequential, data-dependent front-propagation sweep with a parallelized **Level-BFS** strategy.
   * To resolve the seed-frontier conflicts inherent in parallel execution, we introduce a **pinned boundary-seed injection** mechanism (Section 4.2), achieving an $O(|\tilde{D}|)$ computational complexity.
   * This implementation delivers an empirical **$\approx 11\times$ speedup** while maintaining **100% DWC correctness** across all 1,251 BraTS 2021 subjects.
* **Topology Stability Index (TSI):** Evaluates morphological preservation across threshold ranges. A score of $\text{TSI} = 0$ denotes a perfectly stable topology, meaning absolutely no spurious topological structures (such as holes or disconnected islands) are created or destroyed during the immersion pipeline.

This repository contains the source code and experiment reproduction scripts for:
> **DD-FP: Scalable DWC-Consistent Domain-Decomposed Front Propagation for Biomedical Image Analysis** (Preprint, Pattern Recognition, Elsevier, 2026)

---

## Repository layout

```
dd-fp/
├── src/                      # Core library (importable as `src.*`)
│   ├── ddfp/
│   │   ├── __init__.py               # Public API: immersion_pipeline, run_ddfp_2d, get_backend
│   │   ├── gpu_immersion.py          # Level-BFS CUDA kernel + IBI driver (CuPy)
│   │   └── parallel_immersion.py     # CPU fallback (NumPy / SciPy)
│   ├── preprocessing/
│   │   └── preprocessor.py           # NoInterpPreprocessor, NaiveInterpPreprocessor, DDFPPreprocessor
│   └── utils/
│       └── benchmark_utils.py        # verify_dwc, generate_synthetic_volume, time_function, naive_interpolate
│
├── scripts/
│   ├── part_a/                       # §5.1 — DWC correctness & efficiency (Paper Part A)
│   │   ├── exp_a1_correctness.py     # E-NEW-1 · δ violation counts      (Table 2, Fig. 3)
│   │   ├── exp_a2_speedup.py         # E-NEW-3 · wall-clock speedup       (Table 3, Fig. 5)
│   │   ├── exp_a3_delta.py           # E-NEW-2 · K/δ ablation             (Fig. 4, Fig. 6)
│   │   └── exp_a4_scalability.py     # E-NEW-4 · linear scaling           (Fig. 7)
│   │   
│   ├── part_b/                       # §5.2 — Topology-stable analysis (Paper Part B)
│   │   ├── exp_b1_topology_accuracy.py  # DRIVE / CREMI-2D TSI/CC         (Table 4)
│   │   ├── exp_b2_cc_analysis.py        # CC deep analysis, β₀/χ metrics
│   │   ├── exp_b3_brats_all.py          # BraTS-3D full (N=1,251)         (Table 5)
│   │   ├── exp_b4_cremi_3d.py           # CREMI-3D membrane topology
│   │   └── verify_wilcoxon.py           # Reproduce Wilcoxon p<0.001, W=0
│   │
│   ├── ddfp/
│   │   └── experiment_DDFP_all.py    # DD-FP IBI validation (sor_verify_full.json)
│   │
│   └── analysis/                     # Supplementary analysis (TIPS, ε sensitivity)
│       ├── tips.py                   # TIPS (eq:tips): N=20 paper values + N=1,251 full sweep
│       ├── run_epsilon_sensitivity.py# E-NEW-4: ε ∈ {0.1,0.5,1.0,2.0} sweep
│       └── analyse_epsilon.py        # Kruskal-Wallis H=0, p=1.0 (sec:enew4)
│
├── test/
│   ├── conftest.py                   # pytest option: --run-gpu
│   └── test_smoke.py                 # Smoke-test suite (63 tests; GPU tests auto-skipped)
│
├── data/                             # ← NOT committed (see §Data below)
│   ├── BraTS2021/
│   ├── DRIVE/
│   └── CREMI/
│       └── prepare_cremi.py          # CREMI HDF5 → 2-D PNG masks
│
├── results/                          # Output CSVs/JSON; auto-created by scripts
│   ├── part_a/
│   └── part_b/
│
└── requirements.txt
```

---

## Requirements

### Hardware

| Component | Minimum | Used in paper |
|---|---|---|
| GPU | CUDA-capable, 4 GB VRAM | NVIDIA RTX 3080 Ti Laptop (16 GB) |
| CPU | 4 cores | Intel i9-12900H |
| RAM | 16 GB | 32 GB |

The CPU fallback (`parallel_immersion.py`) runs without a GPU but is significantly slower. Part A timing experiments require GPU.

### Software

```
Python       >= 3.9
numpy        >= 1.24
scipy        >= 1.10
Pillow        >= 9.5       # DRIVE/CREMI mask I/O
nibabel      >= 5.0        # BraTS NIfTI loading
h5py         >= 3.8        # CREMI HDF5 loading
pandas       >= 2.0        # verify_wilcoxon.py
matplotlib   >= 3.7        # exp_a4: log-log scaling plot (optional)
PyYAML       >= 6.0        # src config loader
cupy-cudaXXX >= 12.0       # GPU support (optional; match your CUDA version)
```

Install all at once:

```bash
pip install -r requirements.txt
```

For GPU support, install the CuPy build matching your CUDA toolkit ([installation guide](https://docs.cupy.dev/en/stable/install.html)):

```bash
pip install cupy-cuda12x   # example for CUDA 12.x
```

---

## Quickstart

All scripts are run **from the repository root**. `ROOT` is resolved automatically as `Path(__file__).resolve().parent.parent.parent`.

```bash
git clone https://github.com/<your-org>/dd-fp.git
cd dd-fp
pip install -r requirements.txt
```

### Run the smoke-test suite

Verify the installation without any dataset or GPU:

```bash
pytest test/test_smoke.py -v               # GPU tests auto-skipped
pytest test/test_smoke.py -v --run-gpu     # include GPU tests (requires CuPy)
```

Expected: **77 passed**

---

## Core API

```python
from src.ddfp import immersion_pipeline, run_ddfp_2d, get_backend

print(get_backend())   # "gpu" or "cpu"

# 3-D pipeline: uint8 (W, H, D) → float32 (2W-1, 2H-1, 2D-1)
import numpy as np
vol = np.random.randint(0, 256, (64, 64, 64), dtype=np.uint8)
u_dwc = immersion_pipeline(vol)

# 2-D wrapper: float32 (H, W) in [0,1] → float32 (2H-1, 2W-1)
img = np.random.rand(256, 256).astype(np.float32)
u_dwc_2d = run_ddfp_2d(img)
```

Backend selection (GPU vs CPU) is automatic and happens once at import time.

---

## Data preparation

### BraTS 2021

Download the BraTS2021 Dataset (official site [here](https://www.med.upenn.edu/cbica/brats2021/), [kaggle version](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) also available) and extract to `data/BraTS2021/` so that subject directories follow the pattern `BraTS2021_XXXXX/`.

### DRIVE

Download the [DRIVE dataset](https://drive.grand-challenge.org/) and place the manual segmentation masks under `data/DRIVE/training/1st_manual/` (`.gif` or `.png` accepted).

### CREMI

`prepare_cremi.py` can download and convert the CREMI samples automatically:

```bash
# Download samples A/B/C and convert to 2-D PNG masks
python data/CREMI/prepare_cremi.py --download --output-dir data/CREMI/

# Convert existing HDF5 files without downloading
python data/CREMI/prepare_cremi.py --hdf5-dir data/CREMI/raw/ --output-dir data/CREMI/
```

---

## Part A — DWC correctness & efficiency (no dataset required)

All Part A experiments generate synthetic volumes internally and require no external data files.

```bash
python scripts/part_a/exp_a1_correctness.py     # ~2 min  (GPU)
python scripts/part_a/exp_a2_speedup.py         # ~3 min  (GPU recommended)
python scripts/part_a/exp_a3_delta.py           # ~5 min  (GPU)
python scripts/part_a/exp_a4_scalability.py     # ~5 min  (GPU)
```

Results are written to `results/part_a/`.

## Part B — Topology-stable analysis (datasets required)

```bash
# 2D datasets
python scripts/part_b/exp_b1_topology_accuracy.py \
    --drive-gt data/DRIVE/training/1st_manual \
    --cremi-gt data/CREMI/masks --cremi-limit 50 \
    --output results/part_b/exp_b1_results.csv

python scripts/part_b/exp_b2_cc_analysis.py \
    --drive-gt data/DRIVE/training/1st_manual \
    --cremi-gt data/CREMI/masks --cremi-limit 50 \
    --output results/part_b/exp_b2_results.csv

# 3D BraTS (N=1,251 full dataset)
python scripts/part_b/exp_b3_brats_all.py \
    --brats-dir data/BraTS2021

# 3D CREMI (nearest-zoom + δ=8 context padding)
python scripts/part_b/exp_b4_cremi_3d.py \
    --hdf5-dir data/CREMI/raw \
    --n-volumes 3 --n-patches 10 \
    --patch-size 64 128 128 \
    --ddfp-pad 8 \
    --output results/part_b/exp_b4_cremi_3d_results.csv

# Reproduce Wilcoxon signed-rank (p < 0.001, W = 0)
python scripts/part_b/verify_wilcoxon.py \ --dataset drive --csv results/part_b/exp_b1_drive.csv 
```

## DD-FP Validation (Lemma 3+4 & IBI Sweep)

You can replicate the boundary violation removal mechanism (Lemma 3+4) and the Iterative Boundary Injection (IBI) sweep experiments using `scripts/ddfp/experiment_DDFP_all.py`.

* **Verify No-IBI Results Only (E-NEW-1b / Lemma 3+4 Step-Function Test):**
    ```bash
    python scripts/ddfp/experiment_DDFP_all.py --synth_only --skip e2 e3 --K 16 --n_trials 5
    ```
    *Note: The `--n_trials 5` flag automatically runs the experiment 5 times and outputs a `mean ± std` statistical table to account for GPU non-determinism.*

* **Verify IBI Results Only (E-NEW-2 / Practical Range Test):**
    ```bash
    python scripts/ddfp/experiment_DDFP_all.py --synth_only --skip e1b e3 --K 16 --deltas_e2 1 2 --n_trials 5
    ```

* **Full Dataset Validation (BraTS 3-D Full Sweep):**
    ```bash
    python scripts/ddfp/experiment_DDFP_all.py --brats data/BraTS2021 --n 1251 --K 8 --deltas_e1b 0 1 2 3 --deltas_e2 1 2 --max_rounds 16
    ```
---

### Note on GPU Non-Determinism in Synthetic Experiments

When running experiments on synthetic volumes (`synth_gaussian` or `synth_ramp_z`), you may notice that the boundary and total violation counts fluctuate slightly between runs.
* **Cause:** This behavior stems from the parallel reduction and atomic operations inside `front_propagation_gpu`. Minor floating-point rounding variations alter the thread scheduling order, causing continuous float values sitting precisely on the threshold boundary (e.g., near 0.5) to occasionally flip. This cannot be resolved by standard random seeding.
* **Solution & Paper Reporting:** To address this, we introduced the `--n_trials N` argument. The statistics reported in Table 2 (`tab:enew1`) of the paper reflect the `mean ± std` computed over 5 independent runs.
* *Note: This variation **does not occur** when running real **BraTS datasets**, because the underlying MRI data consists of discrete `uint8` integers where intensity values are safely distributed far from the threshold boundaries.*

## ε sensitivity experiments

```bash
# 1. Run sensitivity sweep (ε ∈ {0.1, 0.5, 1.0, 2.0})
python scripts/analysis/run_epsilon_sensitivity.py \
    --brats data/BraTS2021 \
    --out epsilon_sensitivity.json

# 2. Analyse — Kruskal-Wallis H=0, p=1.0
#    Accepts JSON directly (no separate conversion step needed)
python scripts/analysis/analyse_epsilon.py \
    --input epsilon_sensitivity.json \
    --K 16 --delta 1 \
    --out_dir results/analysis
```

---

## Updates & Bug Fixes

- **`exp_a1_correctness.py`**
  - Updated `no_interp_3d`: Changed from `max` to `nearest-zoom` 
  - Fixed `NameError` bug (restored 2 smoke-tests)
     - **Cause:** `csv_path` was used inside `run_a1()` without being defined.
     - **Fix:** Added `csv_path = out_dir / "a1_correctness.csv"` right after `out_dir.mkdir()`.
     - **Addition:** Added missing `import argparse`.

- **`gpu_immersion.py`**
  - Direction A updates
    - Removed deferred hints
    - Implemented separate buffers
    - Fixed the swept direction

- **`exp_a2_speedup.py`** / **`exp_a4_scalability.py`**
  - Completely removed dependency on timing CSVs
     - Deleted the entire `_load_brats_ref_rows()` function.
     - Removed `timing_n100_csv` and `timing_k16_csv` parameters from `run_a2()`.
     - Removed the BraTS reference row append block.
     - Removed `--timing-n100-csv` and `--timing-k16-csv` CLI arguments.
     - Removed `import statistics` and related comment blocks.

- **`exp_b1_topology_accuracy.py`**
  - Removed `--ref-csv` argument and associated logic
     - Deleted `--ref-csv` argparse argument from `_parse()`.
     - Removed `ref_vals` loading logic, drift check block, and conditional column printing from `_print_summary()`.
     - Simplified the `_row()` helper function.

- **`exp_b3_brats_all.py`** / **`exp_b4_cremi_3d.py`**
  - Implemented the `nearest-zoom` fix for `no_interp_3d`

- **`experiment_DDFP_all.py` (CLI Arguments Refactoring & Bug Fix)**
  - **Eliminated Argument Redundancy:** Removed the dead code parameter `--delta_e2`. Explicitly split and renamed the sweeping arguments into `--deltas_e1b` (for E-NEW-1b) and `--deltas_e2` (for E-NEW-2) to avoid user confusion.
  - **Fixed `run_dd_fp_round0` Overlap Bug:** Resolved an issue where expanding `z1_ext` corrupted the inner region assembly by overwriting boundary cell zones. Decoupled the logic into `z1_inner` (assembly boundary) and `z1_ext` (buffer size allocation), successfully achieving exactly 0 violations for $\delta \ge 1$ in `synth_gaussian`.
  - **Rationalized `step_ok` Judgment:** Fixed a false-negative classification where monotonic volumes (`synth_ramp_z`) triggered a `FAIL` status because they naturally yielded 0 violations at $\delta=0$. Aligned the logic with Lemma 4's strict criteria (focusing solely on guaranteeing 0 violations when $\delta \ge 1$).
  - **Added Statistical Logging:** Integrated the `--n_trials` argument to natively compute and display mean, standard deviation, min, and max values across multiple evaluation runs.

---

## Environment used in the paper

| Item | Version |
|---|---|
| OS | Ubuntu 20.04 / WSL2 (Windows 11) |
| Python | 3.10.19 |
| CUDA runtime | 11.8 |
| CUDA driver | 555.97 |
| CuPy | 13.6.0 |
| NumPy | 2.2.6 |
| SciPy | 1.14.1 |
| scikit-image | 0.24.0 |
| nibabel | 5.3.2 |
| matplotlib | 3.9.4 |
| tqdm | 4.67.1 |
| GPU | NVIDIA RTX 3080 Ti Laptop (16 GB GDDR6X) |
| CPU | Intel Core i9-12900H (14 cores, up to 5.0 GHz) |
| System RAM | 24 GB DDR5 |

(Full environment details are provided in Supplement S8 of the paper.)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.