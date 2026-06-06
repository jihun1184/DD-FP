#!/usr/bin/env bash
# scripts/part_a/run_all_part_a.sh
# Run all Part A experiments sequentially.
# Requires: CuPy (for A1 DD-FP, A3, A4), matplotlib, scipy
#
# Usage:
#   cd dd_fp_experiments/
#   bash scripts/part_a/run_all_part_a.sh
#
# To run a single experiment:
#   python scripts/part_a/exp_a1_correctness.py
#   python scripts/part_a/exp_a2_speedup.py
#   python scripts/part_a/exp_a3_delta.py
#   python scripts/part_a/exp_a4_scalability.py

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

echo "========================================"
echo "  DD-FP Part A Experiments"
echo "  root: $ROOT"
echo "========================================"

python scripts/part_a/exp_a1_correctness.py
python scripts/part_a/exp_a2_speedup.py
python scripts/part_a/exp_a3_delta.py
python scripts/part_a/exp_a4_scalability.py

echo ""
echo "All Part A experiments complete."
echo "Results -> $ROOT/results/part_a/"
