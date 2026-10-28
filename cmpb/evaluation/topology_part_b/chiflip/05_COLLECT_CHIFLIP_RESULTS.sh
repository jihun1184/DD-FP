#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${1:?usage: $0 REPO}"
RESULT_DIR="$BUNDLE_ROOT/RESULTS/K16_CHIFLIP_FULLN"
AUDIT="$RESULT_DIR/independent_audit.json"

test -f "$AUDIT" || { echo "Missing $AUDIT; run the full-N chi-flip gate first." >&2; exit 2; }
python - "$AUDIT" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
if not data["execution_integrity"]["pass"]:
    raise SystemExit("Execution-integrity audit did not pass")
print("RESULT_COLLECTION_PREFLIGHT_PASS")
print("DWC_VERDICT =", data["scientific_results"]["dwc_verdict"])
print("RSTAR2_VERDICT =", data["scientific_results"]["rstar2_verdict"])
print("CHI_FLIP_N =", data["scientific_results"]["chi_flip_n"])
print("CHI_FLIP_PCT =", data["scientific_results"]["chi_flip_pct"])
PY

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RETURN_ROOT="$BUNDLE_ROOT/RETURN/DD_FP_P6_K16_CHIFLIP_FULLN_RESULT_${STAMP}"
mkdir -p "$RETURN_ROOT/RESULTS" "$RETURN_ROOT/LOGS" "$RETURN_ROOT/PROVENANCE" "$RETURN_ROOT/REFERENCE"
cp -a "$RESULT_DIR/." "$RETURN_ROOT/RESULTS/"
cp -a "$BUNDLE_ROOT/LOGS/K16_CHIFLIP_FULLN/." "$RETURN_ROOT/LOGS/"
cp "$BUNDLE_ROOT/README_CHIFLIP.md" "$RETURN_ROOT/"
cp "$BUNDLE_ROOT/REFERENCE/brats_ref_chi_1251.csv" "$RETURN_ROOT/REFERENCE/"
cp "$BUNDLE_ROOT/REFERENCE/REFERENCE_SHA256.txt" "$RETURN_ROOT/REFERENCE/"
git -C "$REPO" rev-parse HEAD > "$RETURN_ROOT/PROVENANCE/repo_head.txt"
git -C "$REPO" status --short > "$RETURN_ROOT/PROVENANCE/repo_status_short.txt"
sha256sum "$REPO/scripts/part_b/exp_b3_brats_3d_all.py" > "$RETURN_ROOT/PROVENANCE/exp_b3_sha256.txt"
python - <<'PY' > "$RETURN_ROOT/PROVENANCE/python_cuda_environment.txt"
import platform, sys
import cupy as cp
import nibabel as nib
import numpy as np
import scipy
import skimage
import tqdm
print("python", sys.version.replace("\n", " "))
print("platform", platform.platform())
print("numpy", np.__version__)
print("nibabel", nib.__version__)
print("scipy", scipy.__version__)
print("skimage", skimage.__version__)
print("tqdm", tqdm.__version__)
print("cupy", cp.__version__)
print("cuda_runtime", cp.cuda.runtime.runtimeGetVersion())
print("cuda_devices", cp.cuda.runtime.getDeviceCount())
for i in range(cp.cuda.runtime.getDeviceCount()):
    p = cp.cuda.runtime.getDeviceProperties(i)
    name = p["name"].decode() if isinstance(p["name"], bytes) else p["name"]
    print("gpu", i, name)
PY
nvidia-smi -q > "$RETURN_ROOT/PROVENANCE/nvidia_smi_q.txt"

(
  cd "$RETURN_ROOT"
  find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
)
(
  cd "$BUNDLE_ROOT/RETURN"
  zip -qr "$(basename "$RETURN_ROOT").zip" "$(basename "$RETURN_ROOT")"
  sha256sum "$(basename "$RETURN_ROOT").zip" > "$(basename "$RETURN_ROOT").zip.sha256"
)

echo "Return these two files:"
echo "$RETURN_ROOT.zip"
echo "$RETURN_ROOT.zip.sha256"
