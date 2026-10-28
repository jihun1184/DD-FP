# Figures and tables of the manuscript: generator, input, stored record

Manuscript V9 (Phase 3B final text). Stored records are in `cmpb/data/` (SHA-256 prefixes of 12 hexadecimal characters; full values in `SHA256SUMS.txt`). Scripts are in `cmpb/manuscript_tools/`.

**Status codes**

- **RC (recomputed here)**: the script was executed in the release preparation (CPU, Python 3.13) on the stored record, and the numbers it writes equal the numbers printed in V9. Where a PDF is produced, the PDF bytes differ from the manuscript's file (matplotlib/PDF metadata); only the plotted numbers were compared.
- **GO (generator only)**: a generator for the float exists in this repository, but the stored input is not (or only partly) included, so it was not run.
- **NG (no generator)**: the float is drawn directly in the manuscript source (TikZ / LaTeX tables / hand-written supplement table) or its generator was not located. Its numbers are quoted from the evaluation records named in `CODE_EXPERIMENT_MAPPING.md`.

## 1. Run the recomputation

```bash
cd cmpb/manuscript_tools
mkdir -p ../figsrc ../paper/figures
python make_tables.py                  # -> ../figsrc/mem_main_table.tex, mem_full_table.tex, g5_table.tex
python make_figs.py                    # -> ../paper/figures/fig_runtime.pdf, fig_memory.pdf, fig_numbers.json
python make_fig_frontier_main.py       # -> fig_frontier_main.pdf (+ numbers JSON)
python make_fig_p4_s2.py               # -> fig_frontier_gpu.pdf (+ numbers JSON)
python make_fig_p4_exploratory.py      # -> fig_round_time.pdf
python recompute_drive_cremi2d.py ../data    # 19 checks of the DRIVE / CREMI-2D aggregates against the per-image record
python ../provenance_R23/gen_drive_tables.py ../data/exp_b1_corrected.csv ../figsrc   # DRIVE tables, see note (b)
```

Requirements: `numpy`, `matplotlib` (CPU only).

## 2. Map

| Float | Generator | Stored input (SHA-256 prefix) | Status |
|---|---|---|---|
| Fig. 1, 2, 3, 4, 5; Fig. S1, S2, S3 (schematics) | TikZ in the manuscript source (`figsrc/`, not in this repository) | – | NG |
| Fig. 6 (runtime) | `make_figs.py` | `timing_E6_vs_P6.json` `615a81e80c62`, `subject_table.csv` `7236343381a2`, `tierS_time_ratios_from_frozen_cli.csv` `98a9249791d6`, `rows.jsonl` `585dc6d77f4b` | RC |
| Tab. 2 (memory), Tab. S memory (full) | `make_tables.py` → `mem_main_table.tex`, `mem_full_table.tex` | `p2_memory_recompute.json` `b70f73cfdf0c` (derived tables `p2_derived_tables.json` `e477f25c7042`) | RC (byte-identical `.tex`) |
| Fig. S5 (memory) | `make_figs.py` → `fig_memory.pdf` | `p2_memory_recompute.json` | RC |
| Tab. S11 (time per volume) | `make_tables.py` → `g5_table.tex` | `subject_table.csv` | RC (byte-identical `.tex`) |
| Tab. 3, Tab. S3, S7, S8, S10 (DWC scope, whole-grid and seam-layer records, differing cells) | – (values quoted from evaluation records of tiers V and F) | arrays and records not in this repository | NG; see `CODE_EXPERIMENT_MAPPING.md` |
| Tab. 4, Tab. S12, S15 (Min/Max, operators) | – | record E1 (not in this repository) | NG |
| Tab. S1, S6, S16 (proof steps, assumptions, contributions) | – | `cmpb/theorem_tests/results_*.json` and logs (Table S1) | NG (the tests themselves run: `python cmpb/theorem_tests/r0_verify.py checker`) |
| Fig. 7 (DRIVE DWC mechanism) | `make_fig_dwc_mechanism.py` composes panels from `paper/figure_sources/` | `FIG4_DWC_MECHANISM_SUMMARY_FROZEN.json` `ab37cf6aa7b7`; panel sources not in this repository | GO |
| Fig. 8 (qualitative) | `make_fig_qualitative.py` composes a stored figure; panels from `cmpb/evaluation/topology_part_b/qualitative/` | stored arrays/rasters (not in this repository); provenance `qualitative_cpu_fp_provenance.csv` `f33165fdeab3` | GO |
| Tab. S13, S14 (DRIVE) | `cmpb/provenance_R23/gen_drive_tables.py` (core and per-image table); `recompute_drive_cremi2d.py` (aggregates) | `exp_b1_corrected.csv` `b64e4afafac8` | RC: aggregates 19/19; table bodies identical to V9; see (b) |
| Fig. S6 (DRIVE threshold stability) | not located | `exp_b1_corrected.csv` | NG |
| Fig. S4 (IBI seam, geometric) | not located (geometric record) | – | NG |
| Fig. S7 (frontier activity) | `make_fig_frontier_main.py` | `p4_exploratory/s2/*` | RC (numbers JSON identical) |
| Fig. S8 (worklist control flow) | `make_fig_p4_s2.py` | `p4_exploratory/s2/*` | RC (numbers JSON identical) |
| Fig. S9 (round time) | `make_fig_p4_exploratory.py` | `p4_exploratory/` (`_p_B5.npy` `4487bca9637b`, `_t_B5.npy` `49e521a8cae4`, `_p_B8c.npy` `27aaa1dd5b01`, `_t_B8c.npy` `a5b5cc6c9b8c`, `gpu_rounds_*.csv`) | RC (numbers identical) |
| BraTS χ-flip numbers in the text (335/1,251) | `cmpb/evaluation/topology_part_b/chiflip/` | `brats_chiflip_k16_summary.json` `921202896ac0` (derived summary of the audited full-N run) | GO (summary stored; the run needs BraTS and a GPU) |

## 3. Notes

(a) *What "recomputed" means.* The scripts were run on the stored records in this repository; the printed numbers equal those in V9. This does not re-create the underlying measurements (GPU runs, topology pipelines), which are not repeated.

(b) *DRIVE tables.* The bodies of `drive_core_table.tex` and `drive_perimage_table.tex` generated from `exp_b1_corrected.csv` are identical to the V9 files; only the `\caption` text differs, because the V9 captions were edited by hand after the generator was written (the generator still carries the R2.3 wording). `drive_core_table_S.tex` has the same body as `drive_core_table.tex` (placement option `[H]`); no separate generator was found.

(c) *Derived records.* `brats_chiflip_k16_summary.json` (its `source_sha256_prefix16` field names the three source files) and `qualitative_cpu_fp_provenance.csv` (extracted by `cmpb/provenance/extract_cpu_fp_provenance.py` from the per-case `case_provenance.json` files) were derived from run records outside this repository; machine paths were removed and numbers unchanged.

(d) *Not reproducible from this repository.* The GPU measurements (runtime, memory, whole-grid and seam-layer verification, GPU comparison), the BraTS/DRIVE/CREMI pipelines (need the datasets), the operator comparison with Min/Max, and the qualitative arrays. Their producing code is included where located, and the code state is documented; the results themselves are not claimed to be reproduced here.
