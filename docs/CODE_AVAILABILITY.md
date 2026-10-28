# Code availability statement (draft for the manuscript)

Status: draft. The release tag, commit hash and DOI are fixed after the final manuscript compression (Phase 5) and are left as placeholders. This text replaces the sentence "Source code: the CUDA worklist kernels and the IBI driver are available from the corresponding author on request." in the Data availability section of the V8 manuscript.

## Proposed wording

> **Code availability.** The source code (CPU reference and CUDA worklist kernels, slab decomposition and assembly driver, DWC checkers, evaluation scripts and the numerical tests of the proof steps) is available at https://github.com/jihun1184/DD-FP, release `<TAG>` (commit `<HASH>`, archived at `<DOI>`), under the MIT licence. The repository does not contain the three public datasets, the saved output arrays of the whole-grid verification, or the stored timing, memory and per-image topology records on which the reported tables are based; these are held by the authors. Scripts that regenerate tables and figures from those records are included but need them as input. The raw rows of the operator comparison with local Min/Max interpolation and the three-dimensional BraTS arrays of the topology evaluation were not archived. The GPU experiments were run on one NVIDIA RTX 4090 system (see the supplement for the software environment); the code was not executed on other hardware.

## What the statement relies on

| Statement | Basis in the release candidate |
|---|---|
| CUDA worklist kernels, slab decomposition and assembly driver | `src/ddfp/gpu_native_worklist.py`, `gpu_seed_aware_worklist.py`, `gpu_immersion.py`, `scripts/ddfp/experiment_DDFP_all.py` |
| DWC checkers | `gpu_immersion.verify_dwc_gpu`, `benchmark_utils.verify_dwc`, `cmpb/evaluation/e_runtime_r7_runner/lib/r1_check53.py` |
| Evaluation scripts | `cmpb/evaluation/` (tiers E, G, S, M and the conformance harness) |
| Numerical tests of the proof steps | `cmpb/theorem_tests/` |
| Table/figure scripts need stored records | `cmpb/manuscript_tools/` read `../data/*`, not included |
| Records held by the authors | manuscript, Data availability section (V8) |
| One RTX 4090 system only | supplement, S-C "Environment" |

## Items the authors must settle before the statement is final

1. Release tag, commit hash and DOI (Phase 5).
2. Whether the stored records are to be deposited with the code (for example in a data repository); if so the statement changes.
3. Whether the tier-O C model and the producers of tiers V and F are to be added (see `CODE_EXPERIMENT_MAPPING.md`, section 4); until then the statement must not say that every table can be reproduced from the repository.
4. Operating system and exact package versions for the supplement's environment paragraph.
