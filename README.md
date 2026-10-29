# DD-FP: GPU Front Propagation with Slab Domain Decomposition for Digitally Well-Composed Interpolation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Code accompanying the manuscript

> J. Bae, H. Lee, Y. Bae, H. Choe, J. Hu. *DD-FP: GPU Front Propagation with Slab Domain Decomposition for Digitally Well-Composed Interpolation of Biomedical Volumes.* Unpublished manuscript prepared for submission to *Computer Methods and Programs in Biomedicine* (citation to be added after publication; see [`CITATION.cff`](CITATION.cff)).

> **CMPB submission code: v1.0.1.** The CMPB code is included in public `main`. The formal software release is [`v1.0.1`](https://github.com/jihun1184/DD-FP/releases/tag/v1.0.1); its tag identifies the exact release commit. `v1.0.1` changes documentation and adds retained evidence records and their recomputation script; all source files are identical to `v1.0.0` (`SHA256SUMS.txt`). `v1.0.0` is unchanged. The earlier Pattern Recognition submission code is preserved under `pr-2026-legacy` at `8fdfe6233dfe53f996ed2f09e8810bca53d5323b`. No software DOI has been issued or is claimed. The manuscript and its supplement take precedence over this file.

## 1. Purpose and contribution

Digitally well-composed (DWC) interpolation removes the connectivity-convention ambiguity of an interpolated 3-D image. The construction used here, first-discovery front propagation (FP), is defined by a sequential schedule. The manuscript asks how FP can be run on a GPU and across slab boundaries, and what is and is not preserved. Its contributions, as stated there:

1. **GPU execution and output verification of FP** (GPU-FP, K = 1): a five-property scheduling contract (P1–P5) and a level-synchronous GPU worklist.
2. **Domain decomposition and boundary ownership** (DD-FP, K > 1): z-slabs, a one-voxel halo, window-local zero frames and an ascending-overwrite ownership rule.
3. **A conditional DWC assembly theorem** (Theorem 3.1, proved in Supplement S-A).
4. **A runtime–memory trade-off and a biomedical evaluation** (BraTS 2021, DRIVE, CREMI), including unfavourable findings.

No clinical or downstream benefit is claimed.

## 2. The three operators

| Operator | What it is | Code |
|---|---|---|
| **Sequential FP** | Single-threaded Python reference of first-discovery FP on the expanded lattice (zero frame, start level 0) | `src/ddfp/cpu_fp.py` |
| **GPU-FP (K = 1)** | The worklist kernel of the whole padded volume (frozen kernel, SHA-256 checked at load) | `src/ddfp/gpu_native_worklist.py`, `gpu_seed_aware_worklist.py`, `front_propagation_gpu` in `gpu_immersion.py` |
| **DD-FP (K > 1)** | z-slabs, halo δ = 1, window-local zero frame, ownership assembly. `max_rounds=0` is *direct assembly*; `max_rounds ≥ 1` adds optional seeded rounds (**IBI**) | `scripts/ddfp/experiment_DDFP_all.py` (`run_ibi_v10`) |

GPU-FP (K = 1) and DD-FP (K > 1) are **different operators**. In the evaluated volumes GPU-FP reproduced the sequential output; DD-FP does not (see §3).

## 3. Scope of the claims

- **Output identity (GPU-FP).** In 20 BraTS 2021 volumes the GPU-FP output was identical to the sequential output (SHA-256 of the output arrays, 20/20). This is an observation on those volumes. That the CUDA code realizes P1–P5 is not proved (hypothesis H-GPU).
- **DD-FP output.** At K = 16, δ = 1 direct assembly differed from the sequential output in 0.603–1.54 % of the 70,897,269 expanded cells per volume (IBI final 0.67–1.63 %). Neither is claimed to be better or worse.
- **Theorem 3.1 (conditional; Supplement S-A).** Unseeded direct assembly with z-slabs, δ = 1, zero frame, start level 0, integer-valued input in 0–255 and an admissible P1–P5 run yields a DWC array. The proof is hand-written (developed with AI assistance), not machine-verified and not independently reviewed. **IBI (seeded rounds) is outside the theorem**, as are other halo widths, other decomposition directions and other border conditions. δ = 1 is the analysed setting; no optimality is claimed. No bound R\* ≤ 2 is claimed.
- **Empirical DWC verification.** Zero violations in 282,015,220 face and cube blocks per array, four arrays in each of 20 volumes; an independent CPU checker re-evaluated two arrays. The saved arrays and the script that generated the direct-assembly arrays are not in this release (`docs/CODE_EXPERIMENT_MAPPING.md`). The seam-layer survey of all 1,251 volumes covers xz/yz seam face blocks only. Zero observed violations are not a guarantee beyond the checked arrays.
- **Runtime.** One RTX 4090, 20 BraTS volumes: GPU-FP 14.1 s vs 111.9 s for the unoptimized single-thread Python reference, 8.05× (95 % CI 7.62–8.51); IBI final 14.5 s. This is not a comparison with optimized CPU code. An output-identical active-pile kernel was 7.48× faster than the frozen kernel (separate baseline).
- **Memory.** Slabs solved in sequence on one device, matched slab thickness (about 9.7 layers): allocated peak 0.165–0.233 of the single pass for direct assembly, 0.717–0.742 for IBI final. Effective sample size about one; the single-device implementation retains full-volume arrays; no O(N/K) or out-of-memory-avoidance claim.
- **Optional IBI.** No advantage on any measured quantity; it adds cost.
- **Topology.** DWC removes the connectivity-convention ambiguity of the interpolated field; it does not preserve the topology of the original image. TSI = 0 holds by construction for binary outputs and is no evidence for DD-FP. On DRIVE the output of the arm labelled `ddfp` is more fragmented under 8-connectivity. Cross-arm comparisons involve different input types (gray-level vs binary) and are descriptive.
- **Which algorithm ran in the biomedical records.** DRIVE and CREMI-2D, arm `ddfp`: single-window GPU-FP (K = 1; `DDFPPreprocessor` → `run_ddfp_2d` → one `immersion_pipeline` call; no slabs, no IBI). BraTS χ-flip (26.78 %) and the 1,251-volume seam check: DD-FP, K = 16, δ = 1, IBI final. Held-out reconstruction (BraTS, CREMI): GPU-FP (K = 1) against Min, Max and Linear. CREMI-3D and the 33-subject BraTS CC: the released scripts run one pass (K = 1; CREMI-3D on a patch padded by δ = 8 on the command line), but the stored values are not tied to a recorded run. Theorem 3.1 concerns unseeded direct assembly across slabs; K = 1 results are not evidence for it.

## 4. Data and preprocessing

The datasets are public and are **not** redistributed. `data/` is git-ignored.

| Dataset | Use | Preprocessing described in the manuscript |
|---|---|---|
| BraTS 2021 ([Synapse](https://www.synapse.org/Synapse:syn27046444)) | 20 volumes (whole-grid check, runtime, slab benchmark, GPU comparison); 1,251 (seam-layer survey, χ-flip) | FLAIR, 240×240×155; clipped at the 1st/99th percentile, scaled to integers 0–255; expanded grid 479×479×309 |
| DRIVE ([grand-challenge](https://drive.grand-challenge.org)) | 20 retinal images, 565×584 | binary vessel fields (1st manual annotation) |
| CREMI ([cremi.org](https://cremi.org)) | 50 2-D patches (256×256), 30 3-D patches (64×128×128) | membrane-derived masks; `data/CREMI/prepare_cremi.py` converts the HDF5 files |

## 5. Installation and environment

```bash
git clone https://github.com/jihun1184/DD-FP.git && cd DD-FP && git checkout v1.0.1
python -m pip install -r requirements.txt
python -m pip install cupy-cudaXXx      # GPU only; build matching your CUDA toolkit
```

Reported measurements: NVIDIA RTX 4090 (24,564 MiB), driver 595.99.02, CUDA 13.2, CuPy 14.2.0, NumPy 2.2.6, Python 3.10.22, Intel i9-14900KF. Other configurations were not evaluated. CPU-only parts (sequential FP, CPU checkers, smoke tests, theorem tests) need Python ≥ 3.9 and the packages of `requirements.txt`.

**Line endings.** The frozen kernel is verified by SHA-256 at load (`853bf33a…`). `.gitattributes` prevents newline conversion; do not clone with a setting that rewrites line endings.

## 6. Minimal commands

CPU only (rechecked during finalization on Windows, Python 3.12.14, no CUDA execution):

```python
import numpy as np
from src.ddfp.cpu_fp import build_ispan_cpu, fp_cpu
from src.utils.benchmark_utils import verify_dwc

vol = np.random.default_rng(0).integers(0, 256, (6, 6, 6), dtype=np.uint8)
lo, hi, l_inf = build_ispan_cpu(vol)
u = fp_cpu(lo, hi, l_inf)[1:-1, 1:-1, 1:-1]       # sequential FP, shape (2W-1, 2H-1, 2D-1)
print(verify_dwc(vol, u)["n_violations"])          # 0
```

GPU (calls copied from the evaluation drivers; **not executed in the release preparation: no GPU**):

```python
from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu
from scripts.ddfp.experiment_DDFP_all import run_ibi_v10
U_lo, U_hi, l_inf = build_ispan_gpu(vol)
u_k1 = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)[1:-1, 1:-1, 1:-1]   # GPU-FP, K = 1
u_dd = run_ibi_v10(vol, K=16, delta=1, max_rounds=0)["u_dd"]                      # DD-FP, direct assembly
u_ibi = run_ibi_v10(vol, K=16, delta=1, max_rounds=8)["u_dd"]                     # direct assembly + optional IBI
```

```bash
python -m pytest test/test_smoke.py -q                        # CPU; --run-gpu on a CUDA machine
python cmpb/manuscript_tools/recompute_drive_cremi2d.py cmpb/data   # recompute the DRIVE / CREMI-2D aggregates from the stored per-image record (CPU)
python cmpb/theorem_tests/r0_verify.py checker                # proof-step consistency tests (CPU)
```

## 7. Reproducing the manuscript's results

Per-experiment commands, inputs, stored records and the status of each figure and table are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), [`docs/CODE_EXPERIMENT_MAPPING.md`](docs/CODE_EXPERIMENT_MAPPING.md) and [`docs/FIGURE_TABLE_REPRODUCIBILITY.md`](docs/FIGURE_TABLE_REPRODUCIBILITY.md).
In short: **the DRIVE and CREMI-2D aggregates, the memory and timing tables and part of the figures can be recomputed from the small stored records in `cmpb/data/` (including the held-out slice reconstruction records, `python3 -I cmpb/manuscript_tools/recompute_reconstruction.py cmpb/data`); the GPU experiments need a CUDA machine and the datasets; several records (saved output arrays of the whole-grid verification, per-subject BraTS 3-D topology data, raw rows of the 77-crop Min/Max operator comparison) are not archived and the corresponding numbers cannot be recomputed from this repository.**

## 8. Repository layout

```
src/ddfp/              sequential FP, GPU-FP kernel and wrappers (hash-checked), earlier CPU fallback
src/preprocessing/     preprocessors used by the Part-B topology scripts
src/utils/             CPU DWC check, synthetic volumes, timing helpers
scripts/ddfp/          run_ibi_v10: DD-FP direct assembly and IBI
scripts/part_a/        earlier synthetic-volume experiments (kept because the hash-locked smoke tests load them; not a source of manuscript numbers)
scripts/part_b/        topology experiments (DRIVE/CREMI-2D, BraTS 3-D, CREMI 3-D); verify_wilcoxon.py is exercised by the smoke tests
test/                  smoke tests (hash-locked)
cmpb/evaluation/       evaluation packages (runtime, memory, GPU comparison, conformance, topology producers), copied verbatim
cmpb/theorem_tests/    numerical consistency tests of the proof steps (Supplement S-A.7)
cmpb/manuscript_tools/ figure/table scripts and consistency checks
cmpb/data/          small stored records used by the table/figure scripts
cmpb/provenance/       patch 8fdfe62 → evaluated tree, CPU-FP version comparison, provenance extraction
cmpb/provenance_R23/   earlier numerical-verification scripts (archival; tied to an earlier manuscript version)
docs/                  mapping, reproducibility, code availability, legacy notice, figure/table map
SHA256SUMS.txt         SHA-256 of tracked release files (excluding this manifest itself)
```

## 9. Relationship to earlier versions

- `main` contains the CMPB source tree used for the evaluation (`cmpb/provenance/PATCH_8fdfe62_to_evaluated_tree.diff`) and the evaluation material. The earlier Pattern Recognition code is recoverable from the annotated tag `pr-2026-legacy` (commit `8fdfe6233dfe53f996ed2f09e8810bca53d5323b`). The v1.0.0 release preserves the evaluated source and stored records unchanged; finalization changes documentation and citation metadata only.
- **Sequential reference.** `cpu_fp.py` of `8fdfe62` decoded neighbour indices in an order that does not match the C-order layout of the array. On small random non-cubic 3-D volumes its output differed from the evaluated version and, in 12 of 30 cases, contained DWC violations; cubic and depth-1 (2-D) volumes were identical. Use the version in v1.0.0 or current `main`. See [`docs/CPU_FP_PROVENANCE.md`](docs/CPU_FP_PROVENANCE.md) for which reported results used which version.
- Removed from the earlier README: an unconditional DWC guarantee, δ = 1 being uniquely optimal, a general R\* ≤ 2 bound for IBI, a ≈ 11× algorithmic speed-up, TSI as evidence of topological stability, and the earlier review status. The earlier parameter-sweep scripts (`scripts/analysis/`) were removed from this tree and remain in the Git history and under the tag. Remaining legacy wording in hash-locked or archival files is listed in [`docs/LEGACY_NOTICE.md`](docs/LEGACY_NOTICE.md).

## 10. License and citation

MIT, see [LICENSE](LICENSE). The datasets keep their own licences. Citation metadata: [`CITATION.cff`](CITATION.cff) (version 1.0.1, release date 2026-10-29; accompanying manuscript remains unpublished).
