# DDFP_A1_ACTIVE_FRONTIER (isolated work tree)
K=1 active-frontier prototype, lower tie only. Production code and R2.3 untouched; `frozen/` holds byte-identical copies.
Report: `A0_A1_ACTIVE_FRONTIER_GATE.md`.
CPU-shim evidence (no GPU needed): `python3 -I tests/harness.py --cases 300` ; `python3 -I tests/mutants.py` ; `python3 -I tests/medium_shim.py`
GPU server (NOT yet run): `tests/gpu_small_gate.py --backend cupy`, then `tests/run_gpu_gate.py --n 1`, then `--n 20`.
Requires g++ (shim), numpy, scipy; cupy only on the GPU server.

## Post-G3 additions (2026-10-22)
Report: `G3_CLOSURE_G4_PREP.md`. G3 evidence (RTX 4090, BraTS x20): `evidence/G3/` (read-only, `G3_EVIDENCE.sha256`), recomputation `results/G3/G3_REEVAL.json`, freeze record `results/G3/G3_FREEZE.json`.
- `tests/gate_verdict.py` fail-closed verdict + gather-share rule (`python3 -I tests/test_gate_verdict.py --g3-dir evidence/G3/gate_4090_n20_20261022_180342`); `tests/reeval_g3.py`.
- `tests/run_gpu_gate.py` = v2 (verdict via gate_verdict); the file that produced G3 is `tests/legacy/run_gpu_gate_G3_as_executed.py` (SHA ae2d38dd...). v2 not yet run on a GPU.
- `tests/closure_depth_trace.py` spatial trace of the deepest closure visits (shim-validated on 5 synthetic cases: `results/closure_trace_shim/`); BraTS run pending (server).
- `tests/run_g4_full_accuracy.py` + `tests/subject_io.py` + `tests/data/brats_1251_ids.txt`: streaming 1,251-subject exhaustive accuracy gate (self-test `tests/g4_shim_selftest.py`: 25/25 on the CPU shim; CuPy path not yet run).
- `tests/collect_env.py` environment capture for server runs.
