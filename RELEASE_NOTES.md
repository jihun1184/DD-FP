# Release notes: CMPB release candidate

Branch `cmpb-release-candidate`; base `8fdfe62` (tag `pr-2026-legacy`). No release tag, no DOI. Corresponds to manuscript V9 (Phase 3B final text).

## What changed relative to `8fdfe62`

**Replaced (source tree = the tree used for the evaluation)**
- `src/ddfp/cpu_fp.py`: C-order neighbour decoding (`3ce62b0d` → `8eae554f`). On non-cubic 3-D volumes the earlier file gives different output; see `docs/CPU_FP_PROVENANCE.md`.
- `src/ddfp/gpu_immersion.py`, `scripts/ddfp/experiment_DDFP_all.py`, `test/test_smoke.py`: evaluated versions (`b784a376`, `b6a9b109`, `d1dea404`).
- `README.md`: rewritten for V9; earlier performance and theory claims removed (unconditional DWC, ≈ 11× algorithmic speed-up, general R\* ≤ 2 for IBI, uniquely optimal δ = 1, TSI as evidence of topology preservation).

**Added**
- GPU-FP worklist kernel and wrapper: `src/ddfp/gpu_native_worklist.py` (`853bf33a`, frozen), `gpu_seed_aware_worklist.py` (`c80f56b9`).
- `cmpb/`: evaluation packages (runtime and parity, conformance, memory, controlled slab benchmark, GPU comparison, topology producers), proof-step tests, table/figure scripts, small stored records (`cmpb/data/`), provenance (patch, CPU-FP comparison and extraction).
- `docs/`: `CODE_EXPERIMENT_MAPPING.md`, `REPRODUCIBILITY.md`, `FIGURE_TABLE_REPRODUCIBILITY.md`, `CPU_FP_PROVENANCE.md`, `CODE_AVAILABILITY.md`, `LEGACY_NOTICE.md`.
- `CITATION.cff`, `RELEASE_NOTES.md`, `SHA256SUMS.txt`, `cmpb/FILE_SHA256SUMS.txt`, `.gitattributes`.

**Removed from the tree** (recoverable from history and from the tag `pr-2026-legacy`)
- `scripts/analysis/analyse_epsilon.py`, `run_epsilon_sensitivity.py`, `tips.py`.

**Kept although legacy**: `scripts/part_a/` and `scripts/part_b/verify_wilcoxon.py` (loaded by the hash-locked smoke tests), `src/ddfp/parallel_immersion.py` and the other earlier modules; their legacy wording is listed in `docs/LEGACY_NOTICE.md`.

**`.gitignore`**: dataset locations are ignored at the repository root only; `cmpb/data/` and result JSON/CSV files inside `cmpb/` are tracked.

## Checks made for this candidate (no GPU, no new experiment)
See `docs/REPRODUCIBILITY.md` §1: hash recomputation, smoke tests, compile and static checks, clean-clone and CRLF check, table/figure recomputation from `cmpb/data/`, hygiene scan.

## Not claimed
- Execution of any CUDA path in the preparation of this candidate.
- Reproduction of the GPU measurements, the whole-grid and seam-layer verification, the Min/Max comparison, the BraTS/CREMI topology pipelines or the qualitative arrays (inputs or producers not in the repository; `docs/FIGURE_TABLE_REPRODUCIBILITY.md`).
- That the CUDA code realizes P1–P5 (H-GPU is not proved) or that Theorem SA is machine-verified.

## Requires the author's approval (not done)
Push of the branch, merge into `main`, release tag, GitHub release, archive and DOI.
