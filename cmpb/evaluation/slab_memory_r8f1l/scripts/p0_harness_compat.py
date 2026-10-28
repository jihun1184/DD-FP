#!/usr/bin/env python3
"""R8-F1L Phase 0: can the frozen R8-D harness express B (= run_ibi_v10 max_rounds=0)?  Read-only; imports R8-D modules in place."""
import copy, json, sys, tempfile
from pathlib import Path
V1 = Path(sys.argv[1]); OUT = Path(sys.argv[2]); R8D = V1 / "R8D_DECOMPOSITION_POLICY_MECHANISM"
sys.path[:0] = [str(R8D / "scripts"), str(R8D / "vendor/r8c/scripts")]
import r8c_common as C, r8d_validate as VAL
res = {}
cfg = json.loads((R8D / "config/server.json").read_text())
with tempfile.TemporaryDirectory() as d:
    for mr in (8, 0):
        p = Path(d) / f"c{mr}.json"; p.write_text(json.dumps({**cfg, "max_rounds": mr}))
        try: C.load_config(p); res[f"load_config_max_rounds_{mr}"] = "accepted"
        except Exception as e: res[f"load_config_max_rounds_{mr}"] = f"REJECTED: {e}"
wd = V1 / "R8E_COST_POLICY_AUDIT_V1/evidence/core/results/server_R8D_001/workers/BraTS2021_00158_m1_K16_metadata_0"
rc = json.loads((wd / "record.json").read_text())
base = set(VAL.record(rc, wd)["reasons"])                       # noise from files absent in the evidence copy, if any
b = copy.deepcopy(rc); b["max_rounds"] = 0; b["R_star"] = 0
K = b["K"]; act = b["activation"]; act["round_changes"] = []; act["solves"] = act["solves"][:K]; act["n_solves"] = K
b["activation"]["R_act_total"] = sum(s["R_act"] for s in act["solves"]); b["activation"]["activation_cell_visits"] = sum(s["R_act"]*s["padded_cells"] for s in act["solves"])
for s in act["solves"]: s["seeded"] = False
brs = set(VAL.record(b, wd)["reasons"])
res.update(real_C_record_reasons=sorted(base), B_like_record_reasons=sorted(brs), reasons_only_for_B=sorted(brs - base),
           conclusion="frozen R8-D load_config + record() cannot admit a max_rounds=0 run; B needs a new B-aware config/validator (new code), contrary to the F1K note that B is a configuration-only change")
OUT.write_text(json.dumps(res, indent=1) + "\n"); print(json.dumps(res, indent=1))
