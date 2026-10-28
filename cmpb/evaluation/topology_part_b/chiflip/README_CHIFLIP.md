# DD-FP P6 K16 full-N χ-flip measurement gate — 2026-09-14

## Purpose

This package extends the canonical `DD_FP_P6_K16_FULLN_GPU_GATE_20260904` execution **without changing the production DD-FP repository or the original frozen runner files**.

The added measurement is only:

- `ref_chi`: frozen subject-level reference Euler characteristic from the direct full-N BraTS baseline remeasurement;
- `ddfp_chi`: Euler characteristic measured immediately from the newly produced corrected `res["u_dd"]`;
- `chi_sign_flip = 1[(ref_chi != 0) and (ddfp_chi * ref_chi < 0)]`.

The K16 production execution remains:

- BraTS 2021 full training population, N=1,251;
- K=16, delta=1, max_rounds=8;
- the canonical `preprocess -> run_ibi_v10 -> count_boundary_violations` path;
- resume-safe atomic per-configuration CSV checkpointing.

No assembled DD-FP volume is saved to disk. χ is measured before `res["u_dd"]` is released.

## Frozen parent

Parent ZIP SHA-256:

```text
52e614e099ca538901c469091c5febfe0dd8490d25624e8516b708c10b571ff7
```

The original package `SHA256SUMS.txt` remains untouched and still verifies all original files.

## χ metric provenance

The patched runner imports only the frozen BraTS topology metric source from the prepared repository:

```text
scripts/part_b/exp_b3_brats_3d_all.py
SHA-256 6be95f2c698d1fc5d3aa841bcf1e7c2d612d6ff9603a826449f3c2fa566e16d3
function topology_metrics_3d
```

The unrelated 9-threshold TSI sweep is disabled (`tsi_3d -> 0.0`) so the added measurement computes the threshold-0.5 topology fields only. On the first newly processed case, the runner also calls the frozen metric with two different legacy `ref_b0` values and fails closed if `chi` or `chi_sign_flip` changes.

## Reference χ freeze

`REFERENCE/brats_ref_chi_1251.csv` contains exactly one `ref_chi` value for each of the 1,251 BraTS subjects. It was reduced from the previously audited direct full-N baseline table:

```text
source brats_baseline_rows.csv
SHA-256 33268ebe1aba9a633f8b7742ecfc32af7571e90cdd612f7250873f16420e5c91
reference CSV SHA-256 6c52fc567e81684aadc9faaaefe4d910269b22515b12a78e8d2b638bdcf6d252
```

Both baseline methods had the same `ref_chi` for every subject; the reduction gate required 2,502 rows, 1,251 unique subjects, two rows per subject, and exactly one unique `ref_chi` per subject.

Before pilot/full-N execution, `preflight_chiflip_reference.py` requires the discovered BraTS subject set and the frozen reference mapping to match exactly at N=1,251.

## Recommended local paths

```bash
REPO=/path/to/DD-FP
BRATS=/path/to/BraTS2021_Training_Data
```

## 1. Extract

```bash
unzip DD_FP_P6_K16_CHIFLIP_FULLN_GATE_20260914.zip
cd DD_FP_P6_K16_CHIFLIP_FULLN_GATE_20260914
```

## 2. Original frozen checks

```bash
sha256sum -c SHA256SUMS.txt
bash RUNS/00_CHECK_ENV.sh
```

Optional but recommended before the experiment:

```bash
bash RUNS/02_RUN_P3_CUDA_GATE.sh "$REPO"
```

Proceed only after:

```text
P3_SERVER_GATE_PASS
```

## 3. Mandatory N=3 χ-flip pilot

```bash
bash RUNS/03_RUN_K16_CHIFLIP_PILOT_N3.sh \
  "$REPO" \
  "$BRATS" \
  0
```

Required execution gates include:

```text
P6_K16_CHIFLIP_REFERENCE_PREFLIGHT_PASS subjects=1251
P6_K16_CHIFLIP_EXECUTION_INTEGRITY_PASS
P6_K16_CHIFLIP_PILOT_N3_COMPLETE
```

The DWC, R*=2, and χ-flip values are scientific observations. Do not selectively rerun subjects because of an unexpected scientific result.

## 4. Full N=1,251

One GPU:

```bash
bash RUNS/04_RUN_K16_CHIFLIP_FULLN_MULTI_GPU.sh \
  "$REPO" \
  "$BRATS" \
  0
```

Four GPUs if available:

```bash
bash RUNS/04_RUN_K16_CHIFLIP_FULLN_MULTI_GPU.sh \
  "$REPO" \
  "$BRATS" \
  0,1,2,3
```

If interrupted, rerun the **same command with the same GPU count and IDs**. Each shard is resume-safe. Do not switch an existing one-shard checkpoint into a four-shard layout or vice versa.

Final success marker:

```text
FULL_N_BRATS_DDFP_CHI_FLIP_READY
```

The main merged outputs are:

```text
RESULTS/K16_CHIFLIP_FULLN/brats_1251_k16_d1_chiflip_merged.csv
RESULTS/K16_CHIFLIP_FULLN/brats_1251_k16_d1_chiflip_summary.json
RESULTS/K16_CHIFLIP_FULLN/independent_audit.json
```

`independent_audit.json` reports, among other fields:

```text
scientific_results.chi_flip_n
scientific_results.chi_flip_pct
scientific_results.ref_chi_zero_n
execution_integrity.checks.chi_flip_formula_exact
execution_integrity.checks.reference_chi_values_exact
```

## 5. Package the result

```bash
bash RUNS/05_COLLECT_CHIFLIP_RESULTS.sh "$REPO"
```

Return the generated ZIP and `.sha256` sidecar under `RETURN/`.

## Important scope boundary

This χ-flip rerun does **not** alter or reopen the already frozen structural `b0-consistency=1.000` statement. It supplies only the previously missing direct full-N BraTS DD-FP χ-flip measurement. `time_s` remains execution provenance only and must not be used as a runtime/TIPS benchmark.
