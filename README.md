# DD-FP: GPU Front Propagation with Slab Domain Decomposition for Digitally Well-Composed Interpolation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Code accompanying the manuscript

> J. Bae, H. Lee, Y. Bae, H. Choe, J. Hu. *DD-FP: GPU Front Propagation with Slab Domain Decomposition for Digitally Well-Composed Interpolation of Biomedical Volumes.* Manuscript prepared for *Computer Methods and Programs in Biomedicine* (not yet published; citation to be added).

> **Status.** This is a release **candidate** (branch `cmpb-release-candidate`). It is not tagged and has no DOI. The earlier code state submitted with the previous manuscript is preserved at commit `8fdfe62` on `main` (see [Relationship to earlier versions](#relationship-to-earlier-versions)). Statements below are limited to what the manuscript states; the manuscript and its supplement take precedence.

## What the repository contains

| Component | What it is | Where |
|---|---|---|
| Sequential FP | Single-threaded Python reference of first-discovery front propagation on the expanded lattice | `src/ddfp/cpu_fp.py` |
| GPU-FP (K = 1) | CUDA worklist kernel realizing the five-property scheduling contract (P1–P5), applied to the whole grid; frozen kernel with a hash check | `src/ddfp/gpu_native_worklist.py`, `src/ddfp/gpu_seed_aware_worklist.py`, wrapper `front_propagation_gpu` in `src/ddfp/gpu_immersion.py` |
| DD-FP (K > 1), direct slab assembly | z-slabs, one-voxel halo (δ = 1), window-local zero frame, ownership-based assembly; `max_rounds=0` of `run_ibi_v10` | `scripts/ddfp/experiment_DDFP_all.py` |
| Optional IBI | Seeded refinement rounds r ≥ 1 on top of direct assembly | same function, `max_rounds ≥ 1` |
| DWC checker | Face (xy, xz, yz) and cube block check of a whole array (GPU `verify_dwc_gpu`; vectorised CPU `verify_dwc`; independent CPU checker used for the stored arrays) | `src/ddfp/gpu_immersion.py`, `src/utils/benchmark_utils.py`, `cmpb/evaluation/e_runtime_r7_runner/lib/r1_check53.py` |
| Evaluation drivers of the manuscript (runtime, memory, GPU implementation comparison, conformance, theorem tests) | Stage-by-stage scripts, copied verbatim from the evaluation packages | `cmpb/` |
| Biomedical topology experiments (BraTS / DRIVE / CREMI) and preprocessing | Part-B scripts and preprocessors | `scripts/part_b/`, `src/preprocessing/`, `data/CREMI/prepare_cremi.py` |
| Figure and table scripts of the manuscript | Scripts that regenerate figures/tables from stored records | `cmpb/manuscript_tools/` |

The mapping from each script to the theorem, experiment, table or figure it reproduces, with a confidence label per row, is in [`docs/CODE_EXPERIMENT_MAPPING.md`](docs/CODE_EXPERIMENT_MAPPING.md). Installation, supported environment, commands and limitations are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Scope of the claims

The manuscript reports measurements on the stated samples and one conditional theorem. This code does not extend them.

- **GPU-FP (K = 1) output.** In the 20 evaluated BraTS 2021 volumes the GPU-FP output was identical to the sequential FP output (SHA-256 of the output arrays, 20/20). This is an observation on those volumes. That the CUDA code realizes the scheduling contract P1–P5 is not proved (assumption H-GPU).
- **DD-FP (K > 1) is a different operator from sequential FP.** At K = 16, δ = 1, direct assembly differed from the sequential output in 0.60–1.54 % of the 70,897,269 expanded cells per volume (IBI final: 0.67–1.63 %). Neither is claimed to be better or worse than the sequential output.
- **Conditional DWC theorem (Theorem SA).** Unseeded direct assembly with z-slabs, δ = 1, zero frame, start level 0, integer-valued input in 0–255 and an admissible P1–P5 run yields a digitally well-composed (DWC) array. The proof has not been machine-verified or externally reviewed. The theorem does not cover seeded (IBI) rounds, other halo widths, other decomposition directions or other border conditions. δ = 1 is the setting analysed; no uniqueness or optimality of δ = 1 is claimed. No bound R\* ≤ 2 is claimed for IBI.
- **Empirical DWC verification.** Zero violations in 282,015,220 blocks per array for four arrays in each of 20 volumes (whole-grid face and cube check); an independent CPU checker re-evaluated two direct-assembly arrays. A seam-layer check on all 1,251 BraTS volumes covers xz/yz face blocks at the seams only and is not a full-grid certificate. Zero observed violations are not a guarantee outside the checked arrays.
- **Runtime.** On one RTX 4090 (20 BraTS volumes), GPU-FP was 8.05× faster than the unoptimized single-thread Python reference (95 % CI 7.62–8.51). This is not a comparison with optimized CPU code. An output-equivalent active-set kernel was 7.48× faster than the frozen whole-grid kernel; this is a separate baseline.
- **Memory.** In a controlled benchmark with slabs solved in sequence on one device, the allocated peak of direct assembly was 0.165–0.233 of the single-pass allocated peak at matched slab thickness (about 9.7 layers); IBI final was 0.717–0.742. The effective sample size is about one, the single-device implementation retains full-volume arrays, and the measurements do not support an O(N/K) statement or out-of-memory avoidance.
- **Optional IBI.** No advantage was observed on any measured quantity; it adds cost and is outside Theorem SA.
- **Biomedical topology.** DWC removes the connectivity-convention ambiguity of the interpolated field. It does not preserve the topology of the original image and no downstream or clinical benefit is shown. TSI = 0 holds by construction for any binary output and is not evidence for DD-FP. On DRIVE the DD-FP output is more fragmented under 8-connectivity.

## Installation

```bash
git clone https://github.com/jihun1184/DD-FP.git
cd DD-FP
git checkout cmpb-release-candidate
python -m pip install -r requirements.txt
python -m pip install cupy-cudaXXx      # GPU only; pick the CuPy build matching your CUDA toolkit (see the CuPy installation guide)
```

The CPU parts (sequential FP, CPU checker, unit tests) need no GPU. GPU-FP, DD-FP and the GPU benchmarks need an NVIDIA GPU and CuPy. Reported measurements were obtained with: NVIDIA RTX 4090, driver 595.99.02, CUDA 13.2, CuPy 14.2.0, NumPy 2.2.6, Python 3.10.22, Intel i9-14900KF. Other configurations were not evaluated.

**Line endings.** The frozen kernel is verified by SHA-256 when it is loaded (prefix `853bf33a`). The repository ships a `.gitattributes` that prevents newline conversion; on Windows do not clone with a setting that rewrites line endings, otherwise the kernel is refused at import.

## Minimal commands

CPU only (executed in the release preparation; Python 3.13, no GPU):

```python
import numpy as np
from src.ddfp.cpu_fp import build_ispan_cpu, fp_cpu
from src.utils.benchmark_utils import verify_dwc

vol = np.random.default_rng(0).integers(0, 256, (6, 6, 6), dtype=np.uint8)
lo, hi, l_inf = build_ispan_cpu(vol)               # interval map on the expanded lattice, 1-cell frame
u = fp_cpu(lo, hi, l_inf)[1:-1, 1:-1, 1:-1]       # sequential FP, shape (2W-1, 2H-1, 2D-1)
print(verify_dwc(vol, u)["n_violations"])          # 0
```

GPU (calls copied from the evaluation drivers; **not executed in the release preparation because no GPU was available**):

```python
from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu
from scripts.ddfp.experiment_DDFP_all import run_ibi_v10

U_lo, U_hi, l_inf = build_ispan_gpu(vol)                                  # vol: uint8 (W, H, D)
u_k1 = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)[1:-1, 1:-1, 1:-1]   # GPU-FP, K = 1
res0 = run_ibi_v10(vol, K=16, delta=1, max_rounds=0)["u_dd"]            # DD-FP, direct slab assembly
res1 = run_ibi_v10(vol, K=16, delta=1, max_rounds=8)["u_dd"]            # direct assembly + optional IBI
```

Tests (CPU; GPU tests are skipped without `--run-gpu`):

```bash
python -m pytest test/test_smoke.py -q            # 65 passed, 12 skipped (GPU) in the release-preparation run
```

Stage-level evaluation commands (runtime, memory, GPU implementation comparison, parity) are listed in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Data

BraTS 2021 ([Synapse syn27046444](https://www.synapse.org/Synapse:syn27046444)), DRIVE ([grand-challenge](https://drive.grand-challenge.org)) and CREMI ([cremi.org](https://cremi.org)) are public datasets and are **not** redistributed here. `data/` is git-ignored. `data/CREMI/prepare_cremi.py` converts CREMI HDF5 files to 2-D masks. The stored records behind the manuscript tables (saved arrays, retained evaluation tables, timing and memory records) are held by the authors and are not part of this repository; see `docs/CODE_AVAILABILITY.md`.

## Repository layout

```
src/ddfp/            cpu_fp.py, gpu_immersion.py, gpu_native_worklist.py (frozen), gpu_seed_aware_worklist.py,
                     parallel_immersion.py (earlier CPU fallback)
src/preprocessing/   preprocessors used by the Part-B topology scripts
src/utils/           benchmark_utils.py (CPU DWC check, synthetic volumes, timing helpers)
scripts/ddfp/        experiment_DDFP_all.py (run_ibi_v10: direct assembly and IBI)
scripts/part_a/      earlier synthetic-volume experiments (see legacy notice)
scripts/part_b/      topology experiments on BraTS / DRIVE / CREMI
scripts/analysis/    earlier analysis scripts (see legacy notice)
test/                smoke tests
cmpb/evaluation/     evaluation packages of the manuscript (verbatim copies; see cmpb/README.md)
cmpb/theorem_tests/  numerical consistency tests of the proof steps (Supplement S-A.7)
cmpb/manuscript_tools/  figure and table scripts
cmpb/provenance/     diff between the earlier public state and the evaluated source tree
docs/                code-experiment mapping, reproducibility guide, code availability statement, legacy notice
```

## Relationship to earlier versions

- `main` still points to the earlier public state (commit `8fdfe62`, "Update test"), which accompanied the earlier manuscript. It is unchanged.
- This branch contains that state plus the source tree used for the evaluation in the present manuscript. `cmpb/provenance/PATCH_8fdfe62_to_evaluated_tree.diff` is the exact difference; applying it to `8fdfe62` reproduces `src/` and `test/` of this branch (checked in the release preparation).
- **Sequential reference.** `src/ddfp/cpu_fp.py` of `8fdfe62` decoded neighbour indices in an order that does not match the C-order layout of the array it returns. On small random non-cubic 3-D volumes its output differed from the evaluated version and, in 12 of 30 test cases, contained DWC violations; cubic and depth-1 volumes were identical (`cmpb/provenance/check_cpu_fp_public_vs_evaluated.py`; details in [`docs/LEGACY_NOTICE.md`](docs/LEGACY_NOTICE.md)). Use the version on this branch as the sequential reference; results obtained with the earlier file on non-cubic 3-D volumes are not those of the evaluated reference.
- Statements in the earlier README that are **not** supported by the present manuscript were removed: an unconditional DWC guarantee, δ = 1 being uniquely optimal, a general bound R\* ≤ 2 for IBI, an algorithmic speed-up of about 11×, an interpretation of TSI as evidence of topological stability, and the earlier review status. Some scripts and docstrings in `scripts/part_a/`, `scripts/analysis/` and `src/` still carry such wording from the earlier work; see [`docs/LEGACY_NOTICE.md`](docs/LEGACY_NOTICE.md). Those scripts were not edited.

## License

MIT, see [LICENSE](LICENSE). The datasets keep their own licences and terms of use.
