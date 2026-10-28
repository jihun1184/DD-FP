#!/usr/bin/env python3
"""R8-F1L Phase 0 (CPU, read-only): independent recomputation of R8-D measurement fields.
Reads R8-D worker records (R8E evidence copy). Writes one JSON. Fails loudly on missing/odd input.
Checks: (a) total_blocks from vol_shape, (b) t_global_s plausibility vs lower bounds, (c) CUDA-event vs wall,
(d) known-output hash regression values, (e) NaN/violation fields."""
import glob, json, statistics as st, sys
from pathlib import Path

V1 = Path(sys.argv[1]); OUT = Path(sys.argv[2])
G = str(V1 / "R8E_COST_POLICY_AUDIT_V1/evidence/core/results/server_R8D_001/workers/*/record.json")
recs = []
for f in sorted(glob.glob(G)):
    r = json.load(open(f)); r["_dir"] = Path(f).parent.name; recs.append(r)
if len(recs) != 237: sys.exit(f"expected 237 records, got {len(recs)}")
done = [r for r in recs if r.get("status") == "COMPLETED"]
prod = [r for r in done if r.get("purpose") != "smoke"]     # 225 production-size passes; smoke (tiny, launch-latency bound) reported apart

def blocks(shape):                         # independent of the CUDA kernel: count 2-faces + unit cubes of the expanded lattice
    W, H, D = (2 * s - 1 for s in shape)   # expanded extent
    return (W-1)*(H-1)*D + (W-1)*H*(D-1) + W*(H-1)*(D-1) + (W-1)*(H-1)*(D-1)

# (a) block-count recomputation
mism = []; rows = []
for r in done:
    v = r.get("verification", {})
    if v.get("verify_status") != "DONE": continue
    n = blocks(r["vol_shape"]); rec_n = v["plan"]["quantities"]["total_blocks"]
    if n != rec_n: mism.append((r["_dir"], n, rec_n))
    rows.append(dict(d=r["_dir"], K=r["K"], copies=r["copies"], purpose=r["purpose"], cells=2*r["vol_shape"][0]-1, blocks=n,
                     t_global=v["t_global_s"], t_h2d=v["t_h2d_s"], nan=v["nan_skipped"], viol=v["global_violations"], ok=v["global_dwc_ok"]))
# (b) lower bounds: kernel must at least stream the expanded field once (4 B/cell read) + astype copy (read+write) ; RTX4090 peak ~1008 GB/s
BW = 1008e9
lb = []
for x in rows:
    cells = None
for r in prod:
    v = r.get("verification", {})
    if v.get("verify_status") != "DONE": continue
    sh = r["out_shape"]; cells = sh[0]*sh[1]*sh[2]
    traffic = cells*4*3                       # read for astype, write copy, read by kernel (>= lower bound, ignores neighbour re-reads)
    lb.append(dict(d=r["_dir"], t_global=v["t_global_s"], lower_bound_s=traffic/BW, ratio_over_bound=v["t_global_s"]/(traffic/BW),
                   gblocks_per_s=blocks(r["vol_shape"])/v["t_global_s"]/1e9))
# (c) event vs wall in TIMING passes (S0-5 evidence from historical runs)
ev = [dict(d=r["_dir"], rel=abs(r["cuda_event_envelope_s"]-r["algorithm_wall_s"])/r["algorithm_wall_s"], wall=r["algorithm_wall_s"])
      for r in done if r.get("purpose") == "timing" and "cuda_event_envelope_s" in r]
# t_global vs algorithm time (timing-pass medians per config)
med = {}
for r in done:
    if r.get("purpose") == "timing": med.setdefault((r["subject"], r["copies"], r["K"]), []).append(r["algorithm_wall_s"])
ratio = []
for r in done:
    if r.get("purpose") == "metadata" and r.get("verification", {}).get("verify_status") == "DONE":
        k = (r["subject"], r["copies"], r["K"])
        if k in med: ratio.append(dict(d=r["_dir"], t_global_over_alg=r["verification"]["t_global_s"]/st.median(med[k]),
                                       with_h2d=(r["verification"]["t_global_s"]+r["verification"]["t_h2d_s"])/st.median(med[k])))
# (d) hash regression
want = {"A_K1": "d0ff1d1e224665bd4ac8c78f03eb3a3eb83bdbd9f65b9f48e4bf0c4789b5d9cd",
        "C_K16": "ecf8fb31276946506941a780f689251587220a9cdd97af274c4fbee0605daa0a"}
hx = {}
for r in done:
    if r["subject"].endswith("00158") and r["copies"] == 1 and r["K"] in (1, 16) and r["purpose"] in ("timing", "metadata"):
        hx.setdefault("A_K1" if r["K"] == 1 else "C_K16", set()).add(r["out_sha256"])
q = lambda xs: [min(xs), st.median(xs), max(xs)]
tg_prod = [r["verification"]["t_global_s"] for r in prod if r.get("verification", {}).get("verify_status") == "DONE"]
tg_smoke = [r["verification"]["t_global_s"] for r in done if r.get("purpose") == "smoke"]
by_purpose = {p: q([r["verification"]["t_global_s"] for r in prod if r["purpose"] == p]) for p in ("timing", "profile", "metadata")}
out = dict(n_records=len(recs), n_completed=len(done), n_verified=len(rows),
           blocks_formula_mismatches=mism,
           all_nan_zero=all(x["nan"] == 0 for x in rows), all_viol_zero=all(x["viol"] == 0 and x["ok"] for x in rows),
           n_production=len(prod), n_smoke=len(tg_smoke), t_global_s_production_min_med_max=q(tg_prod), t_global_s_smoke_min_med_max=q(tg_smoke),
           t_global_s_by_purpose_min_med_max=by_purpose,
           t_h2d_s_production_min_med_max=q([r["verification"]["t_h2d_s"] for r in prod]),
           gblocks_per_s_min_med_max=q([x["gblocks_per_s"] for x in lb]),
           min_ratio_t_global_over_bandwidth_lower_bound=min(x["ratio_over_bound"] for x in lb),
           event_vs_wall_rel_diff_timing_n=len(ev), event_vs_wall_rel_diff_min_med_max=q([e["rel"] for e in ev]),
           event_vs_wall_over_2pct=sorted([e["d"] for e in ev if e["rel"] > 0.02]),
           event_minus_wall_abs_s_min_max=[min(r["cuda_event_envelope_s"]-r["algorithm_wall_s"] for r in prod if r["purpose"]=="timing"), max(r["cuda_event_envelope_s"]-r["algorithm_wall_s"] for r in prod if r["purpose"]=="timing")],
           t_global_over_alg_min_med_max=q([x["t_global_over_alg"] for x in ratio]), with_h2d_over_alg_max=max(x["with_h2d"] for x in ratio),
           hash_regression={k: dict(observed=sorted(v), expected=want[k], match=(v == {want[k]})) for k, v in hx.items()})
OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n"); print(json.dumps(out, indent=1, sort_keys=True))
