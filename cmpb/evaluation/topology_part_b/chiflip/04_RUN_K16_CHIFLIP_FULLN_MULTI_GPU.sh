#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${1:?usage: $0 REPO BRATS_ROOT [GPU_IDS e.g. 0 or 0,1,2,3]}"
BRATS="${2:?usage: $0 REPO BRATS_ROOT [GPU_IDS]}"
GPU_IDS="${3:-0}"
REF="$BUNDLE_ROOT/REFERENCE/brats_ref_chi_1251.csv"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
N=${#GPUS[@]}
mkdir -p "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_FULLN" "$BUNDLE_ROOT/LOGS/K16_CHIFLIP_FULLN"
python "$BUNDLE_ROOT/TOOLS/preflight_chiflip_reference.py" --brats-dir "$BRATS" --ref-chi-csv "$REF" --expected-subjects 1251


echo "Launching $N shard(s) on GPUs: ${GPUS[*]}"
pids=()
for ((i=0; i<N; i++)); do
  gpu=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES="$gpu" python "$BUNDLE_ROOT/TOOLS/p6_k16_chiflip_server_shard.py" \
    --repo "$REPO" --brats-dir "$BRATS" --n -1 --seed 42 --max-rounds 8 \
    --configs "16:1" --skip-kappa --resume --num-shards "$N" --shard-index "$i" \
    --ref-chi-csv "$REF" \
    --output "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_FULLN/shard_${i}.csv" \
    --summary "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_FULLN/shard_${i}_summary.json" \
    --failures "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_FULLN/shard_${i}_failures.tsv" \
    > "$BUNDLE_ROOT/LOGS/K16_CHIFLIP_FULLN/shard_${i}.log" 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
if [ "$rc" -ne 0 ]; then
  echo "At least one shard failed. Inspect logs, then rerun this exact command; checkpoints are resume-safe." >&2
  exit 2
fi

python "$BUNDLE_ROOT/TOOLS/merge_k16_chiflip_shards.py" \
  --csv-glob "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_FULLN/shard_*.csv" \
  --failure-glob "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_FULLN/shard_*_failures.tsv" \
  --configs "16:1" --expected-subjects 1251 \
  --output "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_FULLN/brats_1251_k16_d1_chiflip_merged.csv" \
  --summary "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_FULLN/brats_1251_k16_d1_chiflip_summary.json" \
  2>&1 | tee "$BUNDLE_ROOT/LOGS/K16_CHIFLIP_FULLN/merge.log"

python "$BUNDLE_ROOT/TOOLS/audit_k16_chiflip_results.py" \
  --csv "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_FULLN/brats_1251_k16_d1_chiflip_merged.csv" \
  --failure-glob "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_FULLN/shard_*_failures.tsv" \
  --expected-subjects 1251 --ref-chi-csv "$REF" \
  --output "$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_FULLN/independent_audit.json" \
  2>&1 | tee "$BUNDLE_ROOT/LOGS/K16_CHIFLIP_FULLN/audit.log"

echo P6_K16_CHIFLIP_FULLN_COMPLETE
echo FULL_N_BRATS_DDFP_CHI_FLIP_READY
