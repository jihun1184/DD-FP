# Legacy wording in code and scripts

Some source files come from the earlier work (commit `8fdfe62`) and still contain comments, docstrings or output labels that are **not** supported by the present manuscript. They were left unchanged on purpose:

- `src/ddfp/cpu_fp.py`, `src/ddfp/gpu_immersion.py`, `src/ddfp/gpu_native_worklist.py`, `src/ddfp/gpu_seed_aware_worklist.py`, `scripts/ddfp/experiment_DDFP_all.py` and `test/test_smoke.py` are hash-checked by the evaluation drivers (`cmpb/evaluation/e_runtime_r7_runner/r7/r7_common.py`, table `EXPECTED`; prefixes `8eae554f`, `b784a376`, `853bf33a`, `c80f56b9`, `b6a9b109`, `d1dea404`). Editing even a docstring in them makes those checks fail, and the evaluated source tree would no longer be the released one.
- The remaining files are not part of the evidence of the present manuscript, or are copies of the earlier experiment scripts.

Where the wording below conflicts with the README or the manuscript, the README and the manuscript apply.

| Wording in the code | Files | Status relative to the present manuscript |
|---|---|---|
| "DWC guaranteed", "DWC guarantee: Yes (by construction)", "mathematical guarantees are identical", "Self-dual" | `src/ddfp/__init__.py`, `src/ddfp/gpu_immersion.py`, `src/ddfp/parallel_immersion.py`, `src/preprocessing/preprocessor.py`, `scripts/part_b/exp_b2_cc_analysis.py`, `scripts/analysis/tips.py` | Not supported. DWC of an output holds under Theorem SA (unseeded direct assembly, stated assumptions) or is observed on checked arrays. Self-duality is not claimed. |
| "Theorem 1 Step A", "Theorem 2.11", "Theorem 3.1", "Theorem 4.1", "Corollary 3.9", "Lemma 4" | `src/ddfp/cpu_fp.py`, `src/preprocessing/preprocessor.py`, `scripts/analysis/tips.py`, `scripts/part_b/exp_b3_brats_3d_all.py`, `scripts/ddfp/experiment_DDFP_all.py` | These numbers refer to the earlier manuscript. The theorem of the present manuscript is Theorem SA. |
| "R\*=2 on all BraTS subjects", IBI "guarantees" DWC at boundaries | `scripts/ddfp/experiment_DDFP_all.py`, `scripts/part_a/exp_a3_delta.py`, `scripts/analysis/run_epsilon_sensitivity.py` | Not claimed. R\* = 2 is an observation on the evaluated volumes; IBI is outside Theorem SA and showed no advantage. |
| "TIPS", `dwc_guaranteed` flags | `scripts/analysis/tips.py` | Not part of the present manuscript. |
| Sequential-FP rows filled from a theorem when the CPU run fails (`from_theorem` fallback) | `scripts/part_b/exp_b3_brats_3d_all.py`, `scripts/part_b/exp_b4_cremi_3d.py` | Such rows are inferred, not measured. Any result table built from these scripts has to be checked for rows produced by this fallback. |
| "≈ 11× speed-up", "98.5×" | earlier README and comments | Not claimed. The reported runtime comparison is 8.05× against an unoptimized single-thread Python reference. |

`scripts/part_a/` and `scripts/analysis/` are earlier experiments with synthetic volumes and the earlier parameter sweeps. They are kept for traceability and are **not** the source of any number of the present manuscript, unless `docs/CODE_EXPERIMENT_MAPPING.md` says so.

## Known difference: sequential reference of the earlier public commit

In `src/ddfp/cpu_fp.py` of commit `8fdfe62`, `nbs()` decoded the flat index with z as the slowest axis, whereas the array is raveled in C order for shape (Wp, Hp, Dp). The evaluated version (this branch, SHA-256 prefix `8eae554f`) decodes it in C order; the change and a regression test (`test_sequential_fp_matches_coordinate_reference_on_anisotropic_grid`) are in `cmpb/provenance/PATCH_8fdfe62_to_evaluated_tree.diff`.

`cmpb/provenance/check_cpu_fp_public_vs_evaluated.py` runs both versions on small random volumes (CPU, 10 shapes x 3 seeds):

- 17 of 30 cases gave different outputs. All 17 are non-cubic 3-D volumes (18 such cases were run); cubic volumes and depth-1 volumes (2-D inputs) were identical.
- In 12 of the 30 cases the earlier file's output contained DWC violations (1-58 violating blocks per case); the evaluated version had none in any of the 30 cases.

Consequence: on non-cubic 3-D volumes the earlier file is not the sequential reference of the manuscript. Which version produced the sequential-FP arm of earlier BraTS/CREMI-3D result files is not recorded in this repository (tier C in `CODE_EXPERIMENT_MAPPING.md`). This is a small random-volume check, not an analysis of the earlier results.
