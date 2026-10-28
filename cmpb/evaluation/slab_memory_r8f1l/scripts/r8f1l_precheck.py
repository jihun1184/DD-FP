"""S0/S1 in-process prechecks on tiny inputs.  Subcommands:
  b_equiv  (S0-3)  B = be.ibi(vol,K,1,0) reproduces the F1J-recorded GPU B_sha on regenerated F1J inputs (T2 stride + T0 stride)
  negctl   (S0-4)  tampered fields must be caught: DWC violation, zeroed plane, original-voxel change, NaN
  timer    (S0-5)  CUDA-event vs perf_counter on a ~1 s continuous kernel, |rel diff| <= 2 %
  s1       (S1)    >=200 inputs: GPU A/B/C vs CPU references (bitwise), items 1-3 via the independent oracle, A/B/C pairwise differences
Exit 0 only if every check passes; stub backend = plumbing test only (never GPU evidence)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import r8f1l_lib as L
from r8f1l_lib import C, D, O, ROOT, check_array
import r8f1l_compare as CMP

F1J_SEED = 20271701          # frozen in R8-F1J harness


def t2_inputs():             # verbatim logic of r8f1j_harness.t2_plan/inputs_T2 (input regeneration is verified against recorded vol_sha)
    combos = [(f, si, A, K) for f in O.FAMILIES for si, sh in enumerate(O.SHAPES) for A in (2, 3, 4, 16, 256) for K in range(1, min(6, sh[2]) + 1)]
    extra = np.random.default_rng([F1J_SEED, 2]).choice(len(combos), 1500 - len(combos), replace=False)
    plan = [(c, 0) for c in combos] + [(combos[i], 1) for i in sorted(extra)]
    for n, ((f, si, A, K), inst) in enumerate(plan):
        sh = O.SHAPES[si]
        rng = np.random.default_rng([F1J_SEED, 3, O.FAMILIES.index(f), si, A, K, inst])
        yield f"T2_{n:04d}_{f}_{sh[0]}x{sh[1]}x{sh[2]}_A{A}_K{K}_i{inst}", O.gen_volume(f, sh, A, rng), K


def t0_inputs():
    for si, sh in enumerate(O.SHAPES):
        for ai, A in enumerate((2, 4, 256)):
            for i in range(20):
                rng = np.random.default_rng([F1J_SEED, 0, si, ai, i])
                yield f"T0_{sh[0]}x{sh[1]}x{sh[2]}_A{A}_{i:02d}", O.gen_volume("iid", sh, A, rng), 1 + i % min(6, sh[2])


def extra_inputs(n, seed=20271019):
    """IBI-active supplement (NEW, deterministic): dense 16/256-level random fields where the seeded round changes B in ~30 % of cases (F1J T2 inputs: ~2 %)."""
    rng = np.random.default_rng(seed)
    shapes = [(3, 3, 9), (4, 4, 12), (5, 5, 12), (6, 6, 15), (4, 5, 16)]
    for i in range(n):
        sh = shapes[rng.integers(len(shapes))]
        A = int([16, 256][rng.integers(2)])
        K = int(rng.integers(2, min(4, sh[2] // 3) + 1))
        yield f'X1_{i:03d}_{sh[0]}x{sh[1]}x{sh[2]}_A{A}_K{K}', (rng.integers(0, A, sh) * (255 // (A - 1))).astype(np.uint8), K


def backend(c, kind, K=3):
    R7 = D.load_r7(c)
    lib = str(Path(c['r7_runner']) / 'lib')
    return R7.CupyBackend(c['repo'], lib) if kind == 'cupy' else R7.StubBackend(c['repo'], lib, None, K)


def setK(be, K):
    if not be.cuda:
        be.K = K
    return be


def sha_f32(a):
    return L.arr_sha(np.ascontiguousarray(a, np.float32))


def b_equiv(be, a):
    rec = {}
    for tier, gen, f in (('T2', t2_inputs, ROOT / 'config/f1j_T2_cases.jsonl'), ('T0', t0_inputs, ROOT / 'config/f1j_T0_cases.jsonl')):
        rec[tier] = {l['case']: l for l in map(json.loads, f.read_text().splitlines()) if 'case' in l}
    rows, bad = [], []
    for tier, gen, stride in (('T2', t2_inputs, a.stride_t2), ('T0', t0_inputs, a.stride_t0)):
        for i, (name, vol, K) in enumerate(gen()):
            if i % stride:
                continue
            r = rec[tier][name]
            ok_in = L.arr_sha(vol) == r['vol_sha'] if 'vol_sha' in r else None
            u = setK(be, K).ibi(vol, K, 1, 0)
            row = {'case': name, 'K': K, 'input_regenerated_ok': ok_in, 'R_star': u['R_star'], 'B_sha_equal': sha_f32(u['u_dd']) == r['B_sha']}
            rows.append(row)
            if not (row['B_sha_equal'] and ok_in is not False and u['R_star'] == 0):
                bad.append(row)
    return {'n': len(rows), 'n_bad': len(bad), 'bad': bad[:10], 'pass': bool(rows) and not bad, 'scope': 'recorded B_sha = F1J GPU run_ibi_v10(max_rounds=0) float32 bytes'}


def negctl(be, a):
    rng = np.random.default_rng(7)
    vol = rng.integers(0, 256, (5, 5, 12), dtype=np.uint8)
    u = setK(be, 3).ibi(vol, 3, 1, 0)
    base, bz = np.array(u['u_dd'], np.float32), u['boundary_z_orig']
    from r8c_gpu_memory_probe import verify
    v0, c0 = verify(be, base, {'boundary_z_orig': bz}), check_array(base, vol)
    clean = v0['global_dwc_ok'] and v0['nan_skipped'] == 0 and c0['ok']
    res = {'clean_passes': bool(clean)}
    t = base.copy(); t[2, 2, 4], t[3, 3, 4], t[3, 2, 4], t[2, 3, 4] = 255, 255, 0, 0                       # xy face: opposite diagonals disjoint => non-DWC
    v = verify(be, t, {'boundary_z_orig': bz}); res['dwc_violation'] = {'kernel_violations': v['global_violations'], 'cpu_violations': int(sum(be.cpu_check(t).values())), 'detected': v['global_violations'] > 0 and sum(be.cpu_check(t).values()) > 0}
    t = base.copy(); t[:, :, 6] = 0; res['zeroed_plane'] = (lambda c: {'check': c, 'detected': not c['ok']})(check_array(t, vol))
    t = base.copy(); t[4, 4, 4] += 1; res['original_voxel'] = (lambda c: {'diffs': c['original_voxel_differences'], 'detected': c['original_voxel_differences'] > 0 and not c['ok']})(check_array(t, vol))
    t = base.copy(); t[3, 3, 5] = np.nan; v = verify(be, t, {'boundary_z_orig': bz}); c = check_array(t, vol)
    res['nan'] = {'nan_skipped': v['nan_skipped'], 'dwc_ok_flag': v['global_dwc_ok'], 'array_finite': c['finite'], 'detected': (v['nan_skipped'] > 0 or not c['finite']) and not c['ok'],
                  'note': 'verify_dwc_gpu dwc_ok ignores NaN (finding K-1): rejection relies on nan_skipped>0 and check_array.finite'}
    res['pass'] = bool(clean and all(res[k]['detected'] for k in ('dwc_violation', 'zeroed_plane', 'original_voxel', 'nan')))
    return res


def timer(be, a):
    if not be.cuda:
        return {'pass': True, 'status': 'NOT_APPLICABLE (stub backend)'}
    cp = be.cp
    x = cp.random.random(1 << 26, dtype=cp.float32)
    for _ in range(3):
        x = cp.sqrt(x * x + 1.0)
    be.sync()
    rows = []
    for _ in range(3):
        n = 0
        be.sync(); e0, e1 = cp.cuda.Event(), cp.cuda.Event(); e0.record(); t = time.perf_counter()
        while time.perf_counter() - t < 1.0:
            x = cp.sqrt(x * x + 1.0); n += 1
            if n % 20 == 0: be.sync()
        be.sync(); w = time.perf_counter() - t; e1.record(); e1.synchronize()
        ev = cp.cuda.get_elapsed_time(e0, e1) / 1000
        rows.append({'wall_s': w, 'event_s': ev, 'rel_diff': abs(ev - w) / w})
    return {'runs': rows, 'max_rel_diff': max(r['rel_diff'] for r in rows), 'pass': max(r['rel_diff'] for r in rows) <= 0.02, 'criterion': '<= 2 % (F1K S0-5)'}


def s1(be, a):
    import ibi_cpu, cpu_fp, seedref
    cases = [(n, v, K) for i, (n, v, K) in enumerate(t2_inputs()) if i % a.stride == 0 and K >= 2]
    cases = cases[: a.max_cases] + list(extra_inputs(a.extra))
    rows, bad, nBC, nseam_changed = [], [], 0, 0
    for name, vol, K in cases:
        setK(be, K)
        out = {}
        for m, rounds in (('B', 0), ('C', 8)):
            g = be.ibi(vol, K, 1, rounds)
            cpu = ibi_cpu.ibi(vol, K, 1, rounds, tie='lower')
            out[m] = (np.asarray(g['u_dd'], np.float32), g)
            out[m + '_eq'] = bool(np.array_equal(out[m][0].view(np.uint32), np.asarray(cpu['u'], np.float32).view(np.uint32))) and g['R_star'] == cpu['R']
        lo, hi, li = be.prep(vol)
        uA = be.fp_k1(lo, hi, li)[1:-1, 1:-1, 1:-1]
        refA = seedref.seed_aware_cpu_reference(*cpu_fp.build_ispan_cpu(vol), None, 'lower')[0][1:-1, 1:-1, 1:-1]
        a_eq = bool(np.array_equal(np.ascontiguousarray(uA, np.float32).view(np.uint32), np.asarray(refA, np.float32).view(np.uint32)))
        row = {'case': name, 'K': K, 'A_eq_cpu': a_eq, 'B_eq_cpu': out['B_eq'], 'C_eq_cpu': out['C_eq'], 'R_star_C': out['C'][1]['R_star']}
        for m, u in (('A', np.ascontiguousarray(uA, np.float32)), ('B', out['B'][0]), ('C', out['C'][0])):
            oc = O.dwc_count_np(u)
            gk = be.global_check(be.h2d(u))                                    # kernel directly: R8-C verify() refuses degenerate dims (e.g. 1xHxD) via verifier_plan
            seam_ok = m == 'A' or be.seam(u, out['B'][1]['boundary_z_orig']) == 0
            ca = check_array(u, vol)
            row[m + '_items123'] = bool(oc['violations'] == 0 and oc['nan_skipped'] == 0 and ca['ok'] and seam_ok
                                         and (gk['violations'], gk['nan_skipped']) == (oc['violations'], oc['nan_skipped']) and (not be.cuda or gk['total_blocks'] == oc['total_blocks']))
        sB = CMP.pair_stats(out['B'][0], out['C'][0], CMP.seam_distance(vol.shape[2], K))[0]
        sA = CMP.pair_stats(np.ascontiguousarray(uA, np.float32), out['B'][0], CMP.seam_distance(vol.shape[2], K))[0]
        row.update(n_diff_B_C=sB['n_diff'], n_diff_A_B=sA['n_diff'])
        nBC += sB['n_diff'] > 0
        rows.append(row)
        if not all(row[k] for k in ('A_eq_cpu', 'B_eq_cpu', 'C_eq_cpu', 'A_items123', 'B_items123', 'C_items123')):
            bad.append(row)
    return {'n': len(rows), 'n_bad': len(bad), 'bad': bad[:10], 'n_extra_ibi_active_inputs': a.extra, 'n_cases_B_ne_C': nBC, 'n_cases_C_with_ibi_round': sum(r['R_star_C'] >= 1 for r in rows), 'pass': len(rows) >= a.min_cases and not bad,
            'criterion': f'>= {a.min_cases} inputs, GPU==CPU bitwise for A,B,C and items 1-3 hold (F1K S1)'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=('b_equiv', 'negctl', 'timer', 's1'))
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--backend', choices=('cupy', 'stub'), default='cupy')
    ap.add_argument('--stride-t2', type=int, default=30)
    ap.add_argument('--stride-t0', type=int, default=21)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--max-cases', type=int, default=400)
    ap.add_argument('--min-cases', type=int, default=200)
    ap.add_argument('--extra', type=int, default=150)
    a = ap.parse_args()
    c = D.config(a.config)
    t_proc = time.monotonic()
    if a.backend == 'cupy': L.budget_guard('PRE_' + a.cmd)                                    # GPU prechecks draw from the same 9.1 GPU-hour ledger as the stages
    try:
        be = backend(c, a.backend)
        t = time.perf_counter()
        r = {'cmd': a.cmd, 'backend': a.backend, 'cuda_executed': be.cuda, **globals()[a.cmd](be, a)}
        r['wall_s'] = time.perf_counter() - t
    except BaseException as e:
        r = {'cmd': a.cmd, 'pass': False, 'error': f'{type(e).__name__}: {e}'}
    r['own_sources'] = L.own_hashes()
    r['process_wall_s'] = time.monotonic() - t_proc                                           # includes backend init, CPU reference work and failures
    L.ledger_add({'backend': a.backend, 'stage': 'PRE_' + a.cmd, 'run': str(a.out), 'gpu_wall_h': r['process_wall_s'] / 3600, 'wall_h': r['process_wall_s'] / 3600, 'status': 'COMPLETED' if r.get('pass') else 'FAILED'})
    C.writej(a.out, r)
    print(json.dumps({k: v for k, v in r.items() if k not in ('own_sources', 'bad')}, indent=1, default=str))
    return 0 if r.get('pass') else 2


if __name__ == '__main__':
    raise SystemExit(main())
