# P4-R2 Phase E decision (computed by decide_e.py)

- data label: **[GPU]**; run status: FINISHED; prereg SHA-256: `432ac65dbe2eadb8e621be36b0747c057e88fd3a44af464e5f97b3a205928609`
- hypothesis H_E (local scope): on this device/server and these N=3 native BraTS inputs, GPU Exact Worklist C1 (a) is faster than the frozen B5 GPU path and (b) is >= 1.5x faster than the frozen single-thread C engine, with bit-identical output.
- **overall: HOLD** UNSTABLE subjects force HOLD (never GO / NO-GO).

| subject | class | rho_B5 (E1) | rho_C (E2) | rho_cpuC1 | cpu_frozen / cpu_c1 | unstable arms | mem alloc c1/b5 |
|---|---|---|---|---|---|---|---|
| 00158 | GO | 41.123 | 8.239 | 4.789 | 1.721 | - | 0.447 |
| 00228 | UNSTABLE | 41.977 | 7.461 | 4.518 | 1.651 | cpu_frozen | 0.450 |
| 01280 | GO | 41.623 | 8.324 | 4.846 | 1.718 | - | 0.451 |

## Wall time medians [s] (IQR/median)
| subject | cpu_frozen | gpu_b5 | cpu_c1 | gpu_c1 |
|---|---|---|---|---|
| 00158 | 2.617 (0.008) | 13.063 (0.000) | 1.521 (0.017) | 0.318 (0.003) |
| 00228 | 2.709 (0.057) | 15.239 (0.001) | 1.640 (0.212) | 0.363 (0.004) |
| 01280 | 2.761 (0.034) | 13.804 (0.000) | 1.607 (0.018) | 0.332 (0.022) |

This file states a local performance class only. Whether a C1 speed-up is a research contribution is judged separately (Phase F); the speed-up is not attributed to any cause here (see the profile runs).
