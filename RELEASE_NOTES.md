# DD-FP v1.0.0 — CMPB Submission Code

Formal software release for manuscript V9 (Phase 3B), dated 2026-10-28.
Target: the verified documentation-finalized `main` commit identified by tag
[`v1.0.0`](https://github.com/jihun1184/DD-FP/releases/tag/v1.0.0).
The earlier Pattern Recognition code remains at `pr-2026-legacy`
(`8fdfe6233dfe53f996ed2f09e8810bca53d5323b`). No DOI is claimed.

## Included code and changes relative to the legacy version

- Sequential CPU FP: corrected C-order neighbour decoding for non-cubic 3-D volumes in `src/ddfp/cpu_fp.py` (`3ce62b0d…` to `8eae554f…`). See `docs/CPU_FP_PROVENANCE.md` for the limited scope and per-result provenance; this does not establish the provenance of every earlier aggregate.
- GPU-FP: public level-synchronous worklist kernel `src/ddfp/gpu_native_worklist.py` (SHA-256 `853bf33a6bbd9c91f5304a018f0a51ff53e8418360fe1477208cf81df17163ac`), seed-aware wrapper and evaluated routing in `gpu_immersion.py`.
- DD-FP: `scripts/ddfp/experiment_DDFP_all.py` (`run_ibi_v10`), z-slab direct assembly (`max_rounds=0`) and optional seeded IBI rounds. The driver bytes are unchanged from legacy; the evaluated GPU routing differs.
- DWC verification: GPU whole-grid checker, CPU checker and independent checker in `cmpb/evaluation/e_runtime_r7_runner/lib/r1_check53.py`.
- Evaluation and reproducibility: runtime, memory, controlled slab, GPU implementation and topology scripts in `cmpb/evaluation/`; existing numerical proof-step tests in `cmpb/theorem_tests/`; stored-record table/figure scripts in `cmpb/manuscript_tools/`; small records in `cmpb/data/`; mapping, provenance and per-figure/table reproducibility documents in `docs/`.
- Removed legacy parameter-sweep scripts remain recoverable from Git history and `pr-2026-legacy`. Hash-locked legacy comments are documented in `docs/LEGACY_NOTICE.md`.

## Finalization checks actually executed (2026-10-28)

On Windows, Python 3.12.14, installed `requirements.txt` plus test tools in an isolated environment:

- Remote `main` and `cmpb-release-candidate` initially both `44422b357df86790cca98d2d7bd9a869f95bcc47`; legacy tag peeled to the expected legacy commit. Existing local tracked files matched the public tree.
- Baseline root SHA-256 manifest: 261 entries, zero mismatches. CMPB manifest: 223 entries, zero mismatches. Six evaluated hash-checked files matched `r7_common.EXPECTED`; all evaluated sources and stored records remain unchanged.
- `python -m pytest test/test_smoke.py -q`: **65 passed, 12 skipped** (CUDA tests). Initial two failures were temporary-directory permission errors; rerun with a workspace-local temporary directory passed without source changes.
- README CPU example: output shape `(11, 11, 11)`, zero violations.
- `python -m compileall -q src scripts test cmpb data`: passed.
- `pyflakes` 4.0.3: 150 diagnostics on unchanged sources (the earlier preparation recorded 149 with its tool version); includes the previously documented undefined `NoReturn` in a quoted return annotation. Static checking is not reported as warning-free; frozen files were not edited.
- Tracked-file hygiene scan: no credential-pattern hits and no medical-image files; the four `.npy` files are one-dimensional exploratory workload/time records, not image volumes. This is a bounded scan, not a universal privacy guarantee.

Earlier release-preparation checks (Python 3.13 on Linux) are recorded separately in `docs/REPRODUCIBILITY.md`; their reported CPU self-check counts are not new executions in this finalization. No new scientific experiment, proof, CUDA execution or GPU benchmark was performed.

## Reproducible scope and limitations

Stored records support recomputing DRIVE/CREMI-2D aggregates, timing and memory tables and selected plotted quantities. This recomputes summaries, not the underlying GPU measurements. The per-figure/table status is in `docs/FIGURE_TABLE_REPRODUCIBILITY.md`, and experiment-to-code confidence is in `docs/CODE_EXPERIMENT_MAPPING.md`.

BraTS 2021, DRIVE and CREMI datasets are not redistributed. Whole-grid output arrays, qualitative source arrays/rasters, per-subject BraTS 3-D topology data, raw Min/Max comparison rows and complete seam-survey records are not included. Some producers and code-to-record mappings remain unconfirmed. Availability on request is not promised for unarchived or unlocated material.

Theorem SA is conditional on the stated unseeded direct-assembly assumptions; IBI is outside it. H-GPU is not proved; finite zero-violation observations are not a general guarantee. GPU-FP identity was observed on the tested volumes; DD-FP is a different operator. Timing is from one RTX 4090 against an unoptimized single-thread Python reference, and memory results retain their effective-sample-size and full-volume-array limitations. IBI showed no advantage on the measured quantities. No clinical, downstream or topology-preservation benefit is claimed.

These are software-release and evidence-scope notes. Development history is not evidence of scientific novelty or theorem validity. The accompanying manuscript is unpublished. MIT applies to the code; datasets retain their own terms.
