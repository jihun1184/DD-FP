# Held-out slice reconstruction records (BraTS 2021, CREMI)

Derived scalar error metrics only. No image, volume, mask or path is stored, and no
personal data is present (a privacy scan for file-system paths, e-mail addresses and
account names found none). The BraTS subject identifiers are the public
challenge case IDs; the source hashes are SHA-256 of the original downloaded files.
The datasets themselves are **not** redistributed and keep their own licences/terms
(BraTS 2021, CREMI).

| File | Rows | Content | SHA-256 (first 16) |
|---|---|---|---|
| `brats_heldout_slice_per_subject.jsonl` | 1,251 | per-FLAIR-volume MAE/RMSE/bias of GPU-FP, Min, Max, Linear on held-out odd z slices | `d54a9c0d39058c09` |
| `cremi_heldout_slice_per_roi.jsonl` | 9 | the same for 3 fixed 128x128x33 ROIs in each of CREMI samples A, B, C | `5ebd0d7f11c93e6b` |

Producers (preserved unchanged, CUDA machine and datasets required; **not** re-run for this release):
`cmpb/evaluation/reconstruction_minmax/pilot_fp_vs_minmax.py` (BraTS) and
`cmpb/evaluation/reconstruction_minmax/cremi_fp_minmax_p2.py` (CREMI);
`script.sha256` in that directory records the BraTS script hash.

Recomputation of the aggregates quoted in the manuscript (Main Table 4, Supplement S-M):

```
python3 -I cmpb/manuscript_tools/recompute_reconstruction.py cmpb/data
```

The script uses only the Python standard library and runs 21 checks (means, per-subject
win counts 1,251 / 1,175 / 0, the 76 Max-adverse subjects, relative reductions, CREMI means
and counts); the expected result is `0 failures`. This recomputes summaries from the stored
per-unit metrics; it does not repeat the GPU computation, and it does not recompute the pixel-level
reconstruction. The per-subject/ROI file is unrelated to the 77-crop operator comparison (record E1),
whose raw rows remain unarchived.
