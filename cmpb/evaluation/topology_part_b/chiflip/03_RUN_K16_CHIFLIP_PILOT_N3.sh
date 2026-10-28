#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${1:?usage: $0 REPO BRATS_ROOT [GPU_ID]}"
BRATS="${2:?usage: $0 REPO BRATS_ROOT [GPU_ID]}"
GPU_ID="${3:-0}"
REF="$BUNDLE_ROOT/REFERENCE/brats_ref_chi_1251.csv"

mkdir -p "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_PILOT_N3" "$BUNDLE_ROOT/LOGS/K16_CHIFLIP_PILOT_N3"
python "$BUNDLE_ROOT/TOOLS/preflight_chiflip_reference.py" --brats-dir "$BRATS" --ref-chi-csv "$REF" --expected-subjects 1251


CUDA_VISIBLE_DEVICES="$GPU_ID" python "$BUNDLE_ROOT/TOOLS/p6_k16_chiflip_server_shard.py" \
  --repo "$REPO" --brats-dir "$BRATS" --n 3 --seed 42 --max-rounds 8 \
  --configs "16:1" --skip-kappa --resume --num-shards 1 --shard-index 0 \
  --ref-chi-csv "$REF" \
  --output "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_PILOT_N3/shard_0.csv" \
  --summary "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_PILOT_N3/shard_0_summary.json" \
  --failures "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_PILOT_N3/shard_0_failures.tsv" \
  2>&1 | tee "$BUNDLE_ROOT/LOGS/K16_CHIFLIP_PILOT_N3/shard_0.log"

python "$BUNDLE_ROOT/TOOLS/merge_k16_chiflip_shards.py" \
  --csv-glob "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_PILOT_N3/shard_*.csv" \
  --failure-glob "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_PILOT_N3/shard_*_failures.tsv" \
  --configs "16:1" --expected-subjects 3 \
  --output "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_PILOT_N3/brats_n3_k16_d1_chiflip_merged.csv" \
  --summary "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_PILOT_N3/brats_n3_k16_d1_chiflip_summary.json" \
  2>&1 | tee "$BUNDLE_ROOT/LOGS/K16_CHIFLIP_PILOT_N3/merge.log"

python "$BUNDLE_ROOT/TOOLS/audit_k16_chiflip_results.py" \
  --csv "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_PILOT_N3/brats_n3_k16_d1_chiflip_merged.csv" \
  --failure-glob "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_PILOT_N3/shard_*_failures.tsv" \
  --expected-subjects 3 --ref-chi-csv "$REF" \
  --output "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_PILOT_N3/independent_audit.json" \
  2>&1 | tee "$BUNDLE_ROOT/LOGS/K16_CHIFLIP_PILOT_N3/audit.log"

echo P6_K16_CHIFLIP_PILOT_N3_COMPLETE
