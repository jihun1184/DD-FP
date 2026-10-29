# Code availability — DD-FP v1.0.1

Public repository: https://github.com/jihun1184/DD-FP

Formal release: https://github.com/jihun1184/DD-FP/releases/tag/v1.0.1

Version: 1.0.1 (source files identical to v1.0.0, 2026-10-28); release date: 2026-10-29; licence: MIT.
The release tag identifies the exact commit; resolve it with `git rev-parse 'v1.0.1^{commit}'`. The full SHA is also recorded in the GitHub release body and the synchronized manuscript. A commit cannot contain its own commit hash, so this versioned document uses the immutable release tag. No DOI has been issued or is claimed.

The release contains the corrected sequential CPU FP reference; the frozen GPU-FP worklist kernel and wrappers; the z-slab direct-assembly driver and optional IBI; whole-grid DWC checkers; located runtime, memory, GPU-comparison and topology evaluation scripts; existing numerical proof-step tests; and scripts that recompute summaries from the included small stored records.

Paths: `src/ddfp/`, `scripts/ddfp/experiment_DDFP_all.py`, `cmpb/evaluation/`, `cmpb/theorem_tests/`, `cmpb/manuscript_tools/`, `cmpb/data/`.

BraTS 2021, DRIVE and CREMI are not redistributed. Saved whole-grid output arrays, qualitative source arrays/rasters, per-subject BraTS 3-D topology data, raw rows of the 77-crop Min/Max operator comparison and complete seam-survey records are not part of this release. The held-out slice reconstruction metrics (BraTS 1,251 subjects; CREMI 9 ROIs), their producer scripts (`cmpb/evaluation/reconstruction_minmax/`) and a recomputation script are included (`cmpb/data/reconstruction/README.md`). Some producing scripts or their correspondence to retained results remain unconfirmed. No request-based access is promised for material that has not been confirmed available.

`docs/CODE_EXPERIMENT_MAPPING.md` records code-to-result confidence and unlocated producers. `docs/FIGURE_TABLE_REPRODUCIBILITY.md` gives per-figure/table status; `docs/REPRODUCIBILITY.md` gives commands and limits; `docs/CPU_FP_PROVENANCE.md` distinguishes sequential-reference versions. The repository alone does not fully reproduce every figure, table or GPU measurement.

The earlier Pattern Recognition submission code is preserved at `pr-2026-legacy`, commit `8fdfe6233dfe53f996ed2f09e8810bca53d5323b`. Citation metadata is in `CITATION.cff`; the accompanying CMPB manuscript remains unpublished. Scientific claim limits are in README §3 and RELEASE_NOTES.md.
