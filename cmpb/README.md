# `cmpb/` — evaluation material of the CMPB manuscript

This directory holds the scripts used to produce and check the evidence of the manuscript, copied **verbatim** from the evaluation packages, together with the provenance of the source tree. Nothing here was re-implemented. Which script reproduces which theorem, experiment, table or figure is given, with a confidence label, in [`../docs/CODE_EXPERIMENT_MAPPING.md`](../docs/CODE_EXPERIMENT_MAPPING.md).

| Directory | Content | Origin |
|---|---|---|
| `evaluation/e_runtime_r7_runner/` | R7 stage runner (`run7.sh`; stages E2 parity, E1, E6, E3, E4), R3 helper scripts, independent CPU checker, GPU parity fixtures | R7 server runner package |
| `evaluation/conformance_r8f1j/` | Round-0 conformance harness, oracle and CPU reference modules | R8-F1J package |
| `evaluation/memory_r8c/` | Memory benchmark scripts and static memory model | R8-C package |
| `evaluation/slab_memory_r8f1l/` | Controlled slab benchmark (stages S0–S4, gates, statistics, tests) and its vendored R8-D / R8-C / F1J sources (`vendor/`, with `VENDOR_SHA256SUMS.txt`) | R8-F1L package |
| `evaluation/g5_active_set/` | GPU implementation comparison (frozen kernel vs active frontier vs active pile), CPU shim, tests | active-frontier (A1) package |
| `theorem_tests/` | Numerical consistency tests of the proof steps of Theorem SA (Supplement S-A.7), logs and result JSON | proof re-verification package |
| `manuscript_tools/` | Figure and table scripts, DRIVE/CREMI-2D aggregate recomputation, build and consistency checks of the manuscript (V9) | manuscript package |
| `evaluation/topology_part_b/` | Producers of the biomedical topology evidence: DRIVE/CREMI-2D runner (`drive_cremi2d/`), qualitative exporters (`qualitative/`), BraTS χ-flip run (`chiflip/`) | topology evaluation packages |
| `data/` | Small stored records read by the table/figure scripts (per-image topology record, timing, memory, exploratory profiles, derived provenance summaries) | evaluation records (V9 data folder) |
| `provenance/` | `PATCH_8fdfe62_to_evaluated_tree.diff` (earlier public state → evaluated tree), `check_cpu_fp_public_vs_evaluated.py`, `extract_cpu_fp_provenance.py` | R7 runner package; release preparation |
| `provenance_R23/` | Earlier numerical-verification scripts and DRIVE table generator (archival; tied to an earlier manuscript text) | R2.3 package |
| `FILE_SHA256SUMS.txt` | SHA-256 of every file in this directory as committed | generated at release-candidate creation |

## Changes to the copied files

Only machine-specific paths were replaced, in the two configuration files below and in `evaluation/topology_part_b/chiflip/README_CHIFLIP.md`:

- `evaluation/slab_memory_r8f1l/tests/stub_config.json`
- `evaluation/slab_memory_r8f1l/vendor/r8d/config/server.json`

(server user name, home directories and a sandbox mount path became `/path/to/...`). The second file's hash no longer equals the entry in `vendor/VENDOR_SHA256SUMS.txt` (original `72974247…`) for this reason. `evaluation/g5_active_set/frozen/` contains byte-identical copies of `src/ddfp/gpu_native_worklist.py` and `src/ddfp/cpu_fp.py`, which the G5 scripts expect at that path. No code file was edited.

## Not included

Result folders, stored arrays and large records, the git bundle and patch of the sealed-checkout builder, server-specific configuration files, and datasets. See `../docs/REPRODUCIBILITY.md`, section 7.
