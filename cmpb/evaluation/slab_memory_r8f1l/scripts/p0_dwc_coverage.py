#!/usr/bin/env python3
"""R8-F1L Phase 0 item 9: face/cube coverage of dwc_check_all_blocks, and NaN handling (static + executable).
(1) Source fidelity: the decode statements ported below must occur verbatim in the frozen gpu_immersion.py.
(2) Exhaustive bijection: for every shape up to 7x6x8, thread index -> block decodes to each in-range 2-face/unit-cube exactly once.
(3) NaN: nan_skipped is NOT part of dwc_ok in verify_dwc_gpu (source), so acceptance must add nan_skipped==0 (checked in validator)."""
import itertools, json, re, sys
from pathlib import Path
src = Path(sys.argv[1]).read_text(); OUT = Path(sys.argv[2])
must = ["z = t / ((Wp-1)*(Hp-1)); t %= (Wp-1)*(Hp-1);", "y = t / (Wp-1);          x = t % (Wp-1);",
        "y = t / ((Wp-1)*(Dp-1)); t %= (Wp-1)*(Dp-1);", "z = t / (Wp-1);          x = t % (Wp-1);",
        "x = t / ((Hp-1)*(Dp-1)); t %= (Hp-1)*(Dp-1);", "z = t / (Hp-1);          y = t % (Hp-1);",
        "int n_xy = (Wp-1)*(Hp-1)*Dp;", "int n_xz = (Wp-1)*Hp*(Dp-1);",
        "if (any_nan4(v00,v10,v01,v11)) {\n            atomicAdd(&violations[1], 1);", "if (has_nan) { atomicAdd(&violations[1], 1); return; }"]
fid = {m: (m in src) for m in must}
ok_expr = bool(re.search(r"ok\s*=\s*\(viol\s*==\s*0\)", src))
def decode(idx, W, H, D):
    n_xy, n_xz = (W-1)*(H-1)*D, (W-1)*H*(D-1); N2 = n_xy + n_xz + W*(H-1)*(D-1)
    if idx < N2:
        if idx < n_xy:
            t = idx; z = t // ((W-1)*(H-1)); t %= (W-1)*(H-1); y = t // (W-1); x = t % (W-1); return ("xy", x, y, z)
        if idx < n_xy + n_xz:
            t = idx - n_xy; y = t // ((W-1)*(D-1)); t %= (W-1)*(D-1); z = t // (W-1); x = t % (W-1); return ("xz", x, y, z)
        t = idx - n_xy - n_xz; x = t // ((H-1)*(D-1)); t %= (H-1)*(D-1); z = t // (H-1); y = t % (H-1); return ("yz", x, y, z)
    t = idx - N2; z = t // ((W-1)*(H-1)); t %= (W-1)*(H-1); y = t // (W-1); x = t % (W-1); return ("cube", x, y, z)
bad = []; n = 0
for W, H, D in itertools.product(range(2, 8), range(2, 7), range(2, 9)):
    total = (W-1)*(H-1)*D + (W-1)*H*(D-1) + W*(H-1)*(D-1) + (W-1)*(H-1)*(D-1)
    got = [decode(i, W, H, D) for i in range(total)]
    want = ({("xy", x, y, z) for x in range(W-1) for y in range(H-1) for z in range(D)} | {("xz", x, y, z) for x in range(W-1) for y in range(H) for z in range(D-1)}
            | {("yz", x, y, z) for x in range(W) for y in range(H-1) for z in range(D-1)} | {("cube", x, y, z) for x in range(W-1) for y in range(H-1) for z in range(D-1)})
    n += 1
    if len(set(got)) != len(got) or set(got) != want: bad.append((W, H, D))
out = dict(source_fidelity=fid, source_fidelity_all=all(fid.values()), dwc_ok_is_viol_eq_0=ok_expr, shapes_checked=n, bijection_failures=bad,
           note_nan="verify_dwc_gpu: ok=(viol==0); nan_skipped is reported but excluded from dwc_ok => acceptance must require nan_skipped==0 and finite output (r8d_validate.record + check_array do)",
           int32_limit_note="thread index and cell offsets are 32-bit int in the kernel; r8c verifier_plan refuses shapes whose block count/offsets overflow (REFUSED_INT32)")
OUT.write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1)); sys.exit(0 if out["source_fidelity_all"] and not bad and ok_expr else 1)
