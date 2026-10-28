# Sequential CPU FP: which version produced which result

This file records which `cpu_fp.py` produced the sequential-FP arm of each result of the manuscript, how this is known, and how far each statement reaches. No experiment was run for it.

## 1. The two versions

| | Public commit `8fdfe62` | Evaluated tree (this branch) |
|---|---|---|
| `src/ddfp/cpu_fp.py` SHA-256 | prefix `3ce62b0d` | prefix `8eae554f` (full value pinned in `cmpb/evaluation/e_runtime_r7_runner/r7/r7_common.py` and in `r3_scripts/r3_diff_ibi.py`) |
| `nbs()` neighbour decoding | flat index decoded with z as the slowest axis, although arrays of shape (Wp, Hp, Dp) are raveled in C order | C-order decoding, marker string `(nx * Hp + ny) * Dp + nz` (also in `gpu_immersion.py`) |
| Change | – | `cmpb/provenance/PATCH_8fdfe62_to_evaluated_tree.diff`; regression test `test_sequential_fp_matches_coordinate_reference_on_anisotropic_grid` in `test/test_smoke.py` |

Direct comparison (`cmpb/provenance/check_cpu_fp_public_vs_evaluated.py`, executed on CPU, small random volumes; 10 shapes × 3 seeds):

- 17 of 30 cases give different outputs; all 17 are non-cubic 3-D volumes (18 such cases were run). Cubic volumes are identical.
- The public file's output has DWC violations in 12 of 30 cases (1–58 violating blocks); the evaluated file in 0 of 30.
- Depth-1 inputs (2-D images stored as W×H×1): 93 cases (81 + 12) were run with both files, 0 differences.

So the difference can only matter for **non-cubic 3-D inputs** (BraTS 240×240×155, CREMI 3-D 64×128×128). It does not matter for 2-D inputs (DRIVE, CREMI 2-D). This is a statement about the executed small-volume checks; it is not a proof of equivalence for 2-D inputs of every size.

## 2. Evidence levels used below

- **R (record)**: a stored run record, hash gate or provenance JSON names the version. Read, not re-run.
- **C (code)**: the runner source enforces the version (fail-closed check). Read, not re-run.
- **E (executed)**: executed here on CPU.
- **U (unverified)**: no record found.

## 3. Result by result

| Result (manuscript) | Sequential-FP arm? | Version | Basis | Scope of the claim |
|---|---|---|---|---|
| 20/20 identical GPU-FP vs sequential outputs; runtime 111.927 s vs 14.125 s (Sec. 5.2, Fig. 6) | yes (BraTS, non-cubic 3-D) | evaluated, `8eae554f` | C: stage drivers `r7_common.py`, `r7_e6.py`, `r3_diff_ibi.py` refuse to run unless `cpu_fp.py` has the pinned hash. The result folders of these runs are not in this repository | Valid for the evaluated version only. The public file would give a different reference on these volumes. |
| Whole-grid verification, differences between direct assembly and sequential output (Sec. 5.5, Tab. S7, S10) | yes | evaluated (same pins) | C: `r3_scripts/` check `LIB_SHA` of `cpu_fp.py`. The mapping of these scripts to the saved arrays is of confidence B (`CODE_EXPERIMENT_MAPPING.md`) | As above; producer to be confirmed by the authors. |
| DRIVE (20) and CREMI-2D (50) topology aggregates, `exp_b1_corrected.csv` (Sec. 5.7, Tab. S13–S14) | yes (`seq_fp` arm) | corrected core | C: `topology_part_b/drive_cremi2d/run_exp_b1_corrected_r3.py` aborts unless the C-order marker is present in `cpu_fp.py` and `gpu_immersion.py` (absent in `8fdfe62`) and the experiment script equals the base-commit script. E: depth-1 inputs are identical for both versions | Independent of the indexing difference in either case. The 2-D runs ran on the corrected core. |
| Qualitative panels, Fig. 8: 3 BraTS, 20 DRIVE, 30 CREMI-3D cases | yes | evaluated, `8eae554f` | R: `case_provenance.json` of each case (53/53 record `8eae554f`; tabulated in `cmpb/data/qualitative_cpu_fp_provenance.csv`, extracted by `cmpb/provenance/extract_cpu_fp_provenance.py`) | The sequential panels come from the evaluated version. For the 3 BraTS cases the recorded DD-FP driver hash is `4e52fd14`, not the released `b6a9b109`: the DD-FP panels of those cases were not produced by the released driver. They are illustrations, not measurements. |
| BraTS χ-flip of DD-FP (335/1,251 = 26.78 %) and the no-interpolation / bilinear baselines (Sec. 5.7) | **no** | – | R: `p6_k16_chiflip_server_shard.py` and `brats_baseline_summary.json` (baselines only). The frozen source hashes are in the package provenance | Not affected: no sequential FP is involved. The DD-FP arm is a GPU/IBI-final run; its code state is stated in `cmpb/evaluation/topology_part_b/chiflip/README_CHIFLIP.md`. |
| BraTS CC / b0-consistency of DD-FP on 33 subjects; CREMI-3D aggregates (30 patches; DD-FP CC = 0, no-interpolation violation rate 1.2457×10⁻⁵, bilinear χ-flip 50 %) | no sequential-FP value is reported | – | U for the producer of the CREMI-3D aggregates (candidates `scripts/part_b/exp_b3_brats_3d_all.py`, `exp_b4_cremi_3d.py`; mapping confidence B) | The manuscript reports no quantitative sequential-FP value from these runs. If an earlier table contained a sequential-FP row from these scripts, it must be treated as unverified: both scripts contain a `from_theorem` fallback that fills a row from the theorem when the CPU run fails. |
| Min/Max operator comparison (Tab. 4, S15) | FP is a C model of the frozen kernel | – | U: model not in this repository | Unchanged (tier O). |

## 4. Statement for the reader

1. The sequential reference of the manuscript is `src/ddfp/cpu_fp.py` of this branch (`8eae554f`). The `cpu_fp.py` of the public commit `8fdfe62` is not that reference on non-cubic 3-D volumes.
2. All sequential-FP results that the manuscript reports for 3-D volumes (identity, runtime, difference counts, qualitative sequential panels) are tied to `8eae554f` by pins or provenance records. The 2-D results are independent of the difference (corrected core enforced; depth-1 inputs identical).
3. For the CREMI-3D and 33-subject BraTS aggregates the version of the CPU file is not recorded; the manuscript uses no sequential-FP quantity from them, and none is claimed here to come from the evaluated version.
4. This audit read records and runner code and ran small CPU checks. It did not re-run the BraTS or CREMI evaluations.
