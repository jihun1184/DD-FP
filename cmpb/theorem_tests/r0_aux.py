"""Auxiliary exact checks: window consistency (local star = global star on every window cell), star inclusion (Lemma I),
owner-plane coverage and the owner-switch plane set, for all (D,K) with D<=30."""
import numpy as np, json
from r0lib import ispan
from r0_verify import split_domain_1d
rng = np.random.default_rng(5)
res = {"window_consistency_viol": 0, "lemmaI_viol": 0, "owner_switch_planes_wrong": 0, "owned_not_in_window": 0, "cases": 0}
for D in range(2, 31):
    for K in range(1, D + 1):
        W, H = 3, 2
        u = rng.integers(0, 256, (W, H, D)); lo, hi = ispan(u)
        slabs = split_domain_1d(D, K)
        own = [next(k for k, (a, b) in enumerate(slabs) if a <= g // 2 < b) for g in range(2 * D - 1)]
        sw = [g for g in range(2 * D - 2) if own[g] != own[g + 1]]
        want = [2 * slabs[k][1] - 1 for k in range(K - 1)]
        res["owner_switch_planes_wrong"] += int(sw != want)
        for k, (z0, z1) in enumerate(slabs):
            zmin, zmax = max(0, z0 - 1), min(D, z1 + 1)
            lo_w, hi_w = ispan(u[:, :, zmin:zmax])
            res["window_consistency_viol"] += int((lo_w != lo[:, :, 2 * zmin:2 * zmax - 1]).sum() + (hi_w != hi[:, :, 2 * zmin:2 * zmax - 1]).sum())
            owned = [g for g in range(2 * D - 1) if own[g] == k]
            res["owned_not_in_window"] += sum(not (2 * zmin <= g <= 2 * zmax - 2) for g in owned)
            # top plane of the lower window at an internal boundary is 2*z_b
            if k < K - 1:
                res["owned_not_in_window"] += int(2 * zmax - 2 != 2 * z1)
        for zb in want:
            zb = (zb + 1) // 2
            res["lemmaI_viol"] += int((lo[:, :, 2 * zb - 1] > lo[:, :, 2 * zb]).sum() + (hi[:, :, 2 * zb - 1] < hi[:, :, 2 * zb]).sum())
        res["cases"] += 1
print(json.dumps(res)); json.dump(res, open("results_aux.json", "w"))
