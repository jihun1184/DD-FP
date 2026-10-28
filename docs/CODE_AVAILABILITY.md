# Code availability: text for the manuscript and its basis in the repository

Status: for the author's approval. The release tag, commit hash and DOI are placeholders until the release is approved and archived. The text replaces the Code availability paragraph of manuscript V9 (`seg/95_declarations.tex`), which describes the public repository as it was before this release candidate and becomes wrong once the candidate branch is merged.

## Proposed wording

> **Code availability.** The code used for the reported results is available at https://github.com/jihun1184/DD-FP, release `<TAG>` (commit `<HASH>`, archived at `<DOI>`), under the MIT licence: the sequential CPU reference of first-discovery front propagation; the GPU-FP worklist kernel (SHA-256 checked at load) and its wrappers; the z-slab direct-assembly and IBI driver; the whole-grid DWC verifiers; the scripts of the runtime, memory, GPU-implementation and topology evaluations; the numerical tests of the proof steps; and the scripts that recompute the tabulated statistics from the small stored records included in the repository. The three public datasets (BraTS 2021, DRIVE, CREMI) are not redistributed. The saved output arrays of the whole-grid verification, the arrays behind the qualitative figure, the raw rows of the comparison with local Min/Max interpolation and the records of the seam-layer survey are not included and are available from the corresponding author on request; `docs/FIGURE_TABLE_REPRODUCIBILITY.md` states for each figure and table whether it can be recomputed from the repository. The code of the earlier submission is preserved under the tag `pr-2026-legacy`; its sequential reference `cpu_fp.py` indexes non-cubic 3-D volumes differently from the evaluated version (`docs/CPU_FP_PROVENANCE.md`). The GPU measurements were made on one NVIDIA RTX 4090 system (driver 595.99.02, CUDA 13.2, CuPy 14.2.0, NumPy 2.2.6, Python 3.10.22); the GPU code was not run on other hardware.

## What each statement relies on

| Statement | Basis in the release candidate |
|---|---|
| Sequential CPU reference | `src/ddfp/cpu_fp.py` (`8eae554f`) |
| GPU-FP kernel, SHA-256 checked at load | `src/ddfp/gpu_native_worklist.py` (`853bf33a`), `gpu_seed_aware_worklist.py`, `gpu_immersion.py` |
| Direct assembly and IBI driver | `scripts/ddfp/experiment_DDFP_all.py` (`run_ibi_v10`, `b6a9b109`) |
| Whole-grid DWC verifiers | `gpu_immersion.verify_dwc_gpu`, `src/utils/benchmark_utils.verify_dwc`, `cmpb/evaluation/e_runtime_r7_runner/lib/r1_check53.py` |
| Runtime, memory, GPU comparison, topology evaluation scripts | `cmpb/evaluation/` |
| Numerical tests of the proof steps | `cmpb/theorem_tests/` |
| Scripts recomputing tabulated statistics from small records | `cmpb/manuscript_tools/`, records in `cmpb/data/` |
| Arrays, seam-layer survey, Min/Max rows not included | `docs/CODE_EXPERIMENT_MAPPING.md` (tiers V, F, O, qualitative arrays) |
| Earlier code preserved | tag `pr-2026-legacy` (commit `8fdfe62`) |
| One RTX 4090 system only | supplement S-C, environment record `cmpb/data/g5_environment.json` |

## Items the author must settle before the statement is final

1. Release tag, commit hash and DOI (created only after approval).
2. Whether the stored arrays are to be deposited in a data repository; if so the "available on request" sentence changes.
3. Whether the unlocated producers (seam-layer survey, Min/Max C model, CREMI-3D aggregates) are to be added; until then the statement must not say that every table can be reproduced from the repository.
4. The manuscript Data availability section and Supplement S-C must be checked for consistency with this wording (they currently describe the repository before replacement).
