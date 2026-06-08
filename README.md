# DD-FP: Domain-Decomposed Front Propagation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**DD-FP** produces a Digitally Well-Composed (DWC) interpolation field from a grayscale 3-D volume via parallel Level-BFS front propagation and Iterative Boundary Injection (IBI). This repository contains the source code and experiment reproduction scripts for:

> *DD-FP: Scalable DWC-Consistent Domain-Decomposed Front Propagation for Biomedical Image Analysis*

---

## Repository layout

```
dd-fp/
├── src/                              # Core library (importable as `src.*`)
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
│   │   ├── exp_a4_scalability.py     # E-NEW-4 · linear scaling           (Fig. 7)
│   │   └── run_all_part_a.sh         # Run A1–A4 in sequence
│   │
│   ├── part_b/                       # §5.2 — Topology-stable analysis (Paper Part B)
│   │   ├── exp_b1_topology_accuracy.py  # DRIVE / CREMI-2D TSI/CC         (Table 4)
│   │   ├── exp_b2_cc_analysis.py        # CC deep analysis, β₀/χ metrics
│   │   ├── exp_b3_brats_3d_all.py       # BraTS-3D full (N=1,251)         (Table 5)
│   │   ├── exp_b4_cremi_3d.py           # CREMI-3D membrane topology
│   │   └── verify_wilcoxon.py           # Reproduce Wilcoxon p<0.001, W=0
│   │
│   ├── ddfp/
│   │   └── experiment_DDFP_all.py    # DD-FP IBI v10 theory validation (sor_verify_full.json)
│   │
│   ├── walltime/                     # Timing data generation (intermediate CSV/JSON)
│   │   ├── gen_timing_n100.py        # Generates timing_k16.csv  (DD-FP K=16 wall-clock)
│   │   └── measure_cpu_brats.py      # Generates timing_n100.csv (seq_cpu + GPU K=1)
│   │
│   ├── preprocess/                   # One-time dataset preparation (not in scripts.zip)
│   │   └── prepare_cremi.py          # CREMI HDF5 → 2-D PNG masks
│   │
│   └── analysis/                     # Supplementary analysis (TIPS, ε sensitivity, validation)
│       ├── tips.py                   # TIPS (eq:tips): N=20 paper values + N=1,251 full sweep
│       ├── run_epsilon_sensitivity.py# E-NEW-4: ε ∈ {0.1,0.5,1.0,2.0} sweep
│       ├── analyse_epsilon.py        # Kruskal-Wallis H=0, p=1.0 (sec:enew4)
│       └── validate_dwc.py           # NaN fallback H1/H2/H3 (Supplement S8)
│
├── test/
│   ├── conftest.py                   # pytest option: --run-gpu
│   └── test_smoke.py                 # Smoke-test suite (63 tests; GPU tests auto-skipped)
│
├── data/                             # ← NOT committed (see §Data below)
│   ├── BraTS2021/
│   ├── DRIVE/
│   └── CREMI/
│
├── results/                          # Output CSVs/JSON; auto-created by scripts
│   ├── part_a/
│   └── part_b/
│
└── requirements.txt
```

> **`data/` and `results/` are not committed.** Pre-computed timing files (`timing_n100.csv`, `timing_k16.csv`) needed by `exp_a2_speedup.py` and `exp_a4_scalability.py` are provided as release assets.

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

Expected: **77 passed**.

### Part A — DWC correctness & efficiency (no dataset required)

Experiments A1, A3, and A4 generate synthetic volumes internally.
A2 and A4 require pre-computed timing CSVs (provided as release assets, or regenerate via `measure_cpu_brats.py` / `gen_timing_n100.py`).

```bash
# Run all Part A experiments (A1–A4) sequentially.
# A2 and A4 look for timing CSVs at ROOT/timing_n100.csv and ROOT/timing_k16.csv by default.
bash scripts/part_a/run_all_part_a.sh

# Or run individually
python scripts/part_a/exp_a1_correctness.py     # ~2 min  (GPU)

# A2 — reads timing_n100.csv (seq_cpu + gpu1) and timing_k16.csv (DD-FP K=16)
python scripts/part_a/exp_a2_speedup.py \
    --timing-n100-csv timing_n100.csv \
    --timing-k16-csv  timing_k16.csv

# A3 — no CSV required
python scripts/part_a/exp_a3_delta.py           # ~5 min  (GPU)

# A4 — reads timing_n100.csv for the BraTS reference point
python scripts/part_a/exp_a4_scalability.py \
    --timing-n100-csv timing_n100.csv
```

All three arguments default to `ROOT/<filename>` when omitted, so if the CSVs are placed at the repository root the bare `python scripts/part_a/exp_a2_speedup.py` form still works.

Results are written to `results/part_a/`.

### Part B — Topology-stable analysis (datasets required)

```bash
# 2D datasets
python scripts/part_b/exp_b1_topology_accuracy.py \
    --drive-gt data/DRIVE/training/1st_manual \
    --cremi-gt data/CREMI/masks --cremi-limit 50 \
    --output results/part_b/exp_b1_results.csv

# Re-run with comparison against a previous result (optional)
python scripts/part_b/exp_b1_topology_accuracy.py \
    --drive-gt data/DRIVE/training/1st_manual \
    --cremi-gt data/CREMI/masks --cremi-limit 50 \
    --output results/part_b/exp_b1_results.csv \
    --ref-csv results/part_b/exp_b1_results.csv   # previous run for drift check

python scripts/part_b/exp_b2_cc_analysis.py \
    --drive-gt data/DRIVE/training/1st_manual \
    --cremi-gt data/CREMI/masks --cremi-limit 50 \
    --output results/part_b/exp_b2_results.csv

# 3D BraTS (N=1,251 full dataset)
python scripts/part_b/exp_b3_brats_3d_all.py \
    --brats-dir data/BraTS2021

# 3D CREMI
python scripts/part_b/exp_b4_cremi_3d.py \
    --hdf5-dir data/CREMI/raw \
    --output results/part_b/exp_b4_cremi_3d_results.csv

# Reproduce Wilcoxon signed-rank (p < 0.001, W = 0)
python scripts/part_b/verify_wilcoxon.py \
    --csv results/part_b/exp_b1_results.csv
```

---

## Data preparation

### BraTS 2021

Download the **Task 1 training set** from [Kaggle](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) (free registration required). Extract to `data/BraTS2021/` so that subject directories follow the pattern `BraTS2021_XXXXX/`.

### DRIVE

Download the [DRIVE dataset](https://drive.grand-challenge.org/) and place the manual segmentation masks under `data/DRIVE/training/1st_manual/` (`.gif` or `.png` accepted).

### CREMI

`prepare_cremi.py` can download and convert the CREMI samples automatically:

```bash
# Download samples A/B/C and convert to 2-D PNG masks
python scripts/preprocess/prepare_cremi.py \
    --download \
    --output-dir data/CREMI/

# Convert existing HDF5 files without downloading
python scripts/preprocess/prepare_cremi.py \
    --hdf5-dir data/CREMI/raw/ \
    --output-dir data/CREMI/
```

### Generating intermediate timing data

Pre-computed files are provided as release assets. Re-run these only to reproduce timing on different hardware.

```bash
# timing_n100.csv  (seq_cpu + GPU single-pass K=1, N=100 BraTS subjects)
# Required by: exp_a2_speedup.py (--timing-n100-csv), exp_a4_scalability.py (--timing-n100-csv)
python scripts/walltime/measure_cpu_brats.py \
    --brats data/BraTS2021 \
    --n 100

# timing_k16.csv  (DD-FP K=16 wall-clock, N=100 BraTS subjects)
# Required by: exp_a2_speedup.py (--timing-k16-csv)
python scripts/walltime/gen_timing_n100.py \
    --brats-dir data/BraTS2021 \
    --output timing_k16.csv \
    --n-subjects 100 --random-seed 42 --K 16

# sor_verify_full.json  (IBI R* census, N=1,251 BraTS)
python scripts/ddfp/experiment_DDFP_all.py \
    --brats data/BraTS2021 \
    --n 1251 --seed 42
```

### Table 3 + TIPS (sec:enew3)

Wall-clock timing and TIPS share the same N=20 measurement
(`TIPS = N_safe / t_wall`, so both come from the same timed run).
N=20 is sufficient for timing: all volumes are 240×240×155 with CV < 5%.

```bash
# TIPS — N=20, paper values (~15 min, GPU only)
python scripts/analysis/tips.py \
    --brats data/BraTS2021 --n 20 --seed 42 --skip-cpu

# TIPS — full dataset N=1,251 (~87 min, DD-FP + GPU K=1 only)
python scripts/analysis/tips.py \
    --brats data/BraTS2021 --n 1251 --seed 42 --skip-cpu
```

> **Note:** Full-dataset TIPS (N=1,251) extends the paper's N=20 table
> for supplementary reproducibility and does not replace tab:speedup values.

### E-NEW-4 — ε sensitivity (sec:enew4)

```bash
# 1. Run sensitivity sweep (ε ∈ {0.1, 0.5, 1.0, 2.0})
python scripts/analysis/run_epsilon_sensitivity.py \
    --brats data/BraTS2021 \
    --out epsilon_sensitivity.json

# 2. Analyse — Kruskal-Wallis H=0, p=1.0
#    Accepts JSON directly (no separate conversion step needed)
python scripts/analysis/analyse_epsilon.py \
    --input epsilon_sensitivity.json \
    --K 4 --delta 1 \
    --out_dir results/analysis
```

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

## Reproducibility notes

- All experiments that sample from BraTS use `seed=42` (passed via `--random-seed`).
- The N=20 timing subsample in Table 3 is the first 20 subjects drawn by that seed from the 1,251-subject training set.
- `verify_wilcoxon.py` reads a pre-existing results CSV and does not re-run the preprocessing pipeline, so it can be executed offline once Part B CSVs exist.
- `exp_a2_speedup.py` reads pre-computed timing CSVs (`timing_n100.csv`, `timing_k16.csv`) rather than re-measuring; to reproduce on different hardware regenerate them with `measure_cpu_brats.py` (for `timing_n100.csv`) and `gen_timing_n100.py --K 16` (for `timing_k16.csv`). Pass the paths explicitly via `--timing-n100-csv` and `--timing-k16-csv`, or place them at the repository root to use the defaults.

---

## Environment used in the paper

| Item | Version |
|---|---|
| OS | Ubuntu 20.04 / WSL2 |
| Python | 3.10 |
| CUDA | 11.8 |
| CuPy | 12.2 |
| NumPy | 1.26 |
| SciPy | 1.11 |
| GPU | NVIDIA RTX 3080 Ti Laptop (16 GB) |
| CPU | Intel i9-12900H |

Full environment details are provided in Supplement S8 of the paper.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.