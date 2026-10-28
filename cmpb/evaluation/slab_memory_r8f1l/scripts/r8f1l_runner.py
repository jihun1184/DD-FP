"""R8-F1L controller: sequential fresh-process jobs (no parallel GPU jobs: timing interference), stop on first defect, per-stage wall rule.
  python -B scripts/r8f1l_runner.py --stage S0|S2|S3|S3b|S4 --run results/<name> --config <server.json> [--backend cupy|stub] [--dry-plan] [--resume]
S3/S3b/S4 start only if <run>/GATE_S2_PASS.json exists (written by r8f1l_gate.py).  Defects are preserved, never retried."""
import argparse
import json
import os
import random
import subprocess
import sys
import shutil
import time
from pathlib import Path

import numpy as np
import r8f1l_compare as CMP
import r8f1l_lib as L
from r8f1l_lib import C, D, ROOT

SUBJ = 'BraTS2021_00158'
from r8f1l_lib import EST_H, CAP_H, WALL_RULE, CAP_GB, ledger_add, ledger_read                    # one budget ledger for runner and prechecks
KLADDER = [2, 4, 8, 16, 32, 64, 128]
REPS = {'S0': 3, 'S2': 5, 'S3': 5, 'S3b': 3, 'S4': 3}                                      # timing repetitions (F1K §2.6; S2 follows §2.6, see freeze P-1)


def job(stage, spec, method, K, purpose, rep, **kw):
    return dict(id=f'{stage}_{spec.replace(":", "-")}_{method}{K if method != "A" else ""}_{purpose}_{rep}', stage=stage, input=spec, method=method,
                K=1 if method == 'A' else K, purpose=purpose, rep=rep, **kw)


def passes(stage, spec, method, K, profile=False, **kw):
    js = [job(stage, spec, method, K, 'timing', i, **kw) for i in range(REPS[stage])]
    js.append(job(stage, spec, method, K, 'metadata', 0, **kw))
    if profile:
        js.append(job(stage, spec, method, K, 'profile', 0, **kw))
    return js


def plan_S0():
    js = [job('S0', 'smoke', m, k, 'smoke', r, cpu_ref=True) for (m, k) in (('A', 1), ('B', 8), ('C', 8)) for r in (0, 1)]
    b1 = f'brats:{SUBJ}:1'
    js += [job('S0', b1, 'A', 1, 'timing', r, regression=True) for r in range(3)] + [job('S0', b1, 'A', 1, 'metadata', 0, regression=True)]
    js += [job('S0', b1, 'B', 16, 'timing', 0), job('S0', b1, 'B', 16, 'metadata', 0), job('S0', b1, 'B', 16, 'profile', 0)]
    js += [job('S0', b1, 'C', 16, 'timing', 0, regression=True), job('S0', b1, 'C', 16, 'metadata', 0, regression=True)]
    return js


def plan_S2():
    ins = {f'{f}{d}': f'synth:{f}:128x128x{d}:256:{11 + i}' for i, (f, d) in enumerate((f, d) for f in ('randomwalk', 'zlayer_noise') for d in (64, 192))}
    js = []
    for name, spec in ins.items():
        for m, K in [('A', 1)] + [(m, K) for m in 'BC' for K in (4, 16)]:
            js += passes('S2', spec, m, K, profile=True)
    sweep = ins['randomwalk192']
    for m in 'BC':
        for K in (2, 8, 32):                                                              # K=4,16 already in the primary set (deduplicated)
            js += passes('S2', sweep, m, K)
    return js


def ladder_around(k, w=1):
    i = KLADDER.index(k)
    return [KLADDER[j] for j in range(max(0, i - w), min(len(KLADDER), i + w + 1))]


def plan_S3(kstar=None):
    kstar = kstar or {1: 16, 2: 32, 4: 64}                                                # F1K: R8-D observed optimum, fixed in advance
    js = []
    for m_ in (1, 2, 4):
        spec = f'brats:{SUBJ}:{m_}'
        js += passes('S3', spec, 'A', 1, regression=(m_ == 1))
        for meth in 'BC':
            for K in (kstar[m_] // 2, kstar[m_], kstar[m_] * 2):
                js += passes('S3', spec, meth, K, regression=(m_ == 1 and meth == 'C' and K == 16))
    return js


def plan_S3b(kstar_B, kstar_C):
    """Main analysis = B and C at the SAME K (union of the two +-1 ladders), so the IBI effect is not confounded with K. Each method's own K* is the secondary analysis (stats.h1_secondary)."""
    Ks = sorted(set(ladder_around(kstar_B)) | set(ladder_around(kstar_C)))
    js = []
    for sid in ('BraTS2021_01363', 'BraTS2021_01628'):
        spec = f'brats:{sid}:1'
        js += passes('S3b', spec, 'A', 1)
        for meth in 'BC':
            for K in Ks:
                js += passes('S3b', spec, meth, K)
    return js


def plan_S4():
    specs = [f'{base}:{s}:{p}' for base in ('bratsinj:' + SUBJ, 'synthinj') for s in L.INJ_KINDS for p in L.INJ_POS]
    specs = [f'bratsinj:{SUBJ}:{s}:{p}' for s in L.INJ_KINDS for p in L.INJ_POS]           # F1K: 3 positions x 2 structures = 6 volumes (BraTS injection; synthetic variant optional)
    js = []
    for spec in specs:
        for meth, K in (('A', 1), ('B', 16), ('C', 16)):
            js += passes('S4', spec, meth, K)
    return js


def make_plan(stage, run=None):
    if stage == 'S0': js = plan_S0()
    elif stage == 'S2': js = plan_S2()
    elif stage == 'S3': js = plan_S3()
    elif stage == 'S4': js = plan_S4()
    elif stage == 'S3b':
        k = C.readj(run / 'KSTAR.json')                                                    # written by r8f1l_stats.py from S3 timing passes (rule fixed before data)
        js = plan_S3b(k['B'], k['C'])
    else: raise SystemExit('unknown stage')
    rng = random.Random(20271018)                                                           # shuffle within replicate blocks (R8-D practice): no config owns a fixed order slot
    key = lambda j: (j['purpose'], j['rep'])
    out = []
    for kk in sorted({key(j) for j in js}, key=lambda t: (t[0] != 'smoke', t)):
        blk = [j for j in js if key(j) == kk]
        if kk[0] == 'metadata':                                                              # grouped per input (A first, then K ascending, B before C): lets outputs be compared and discarded early
            blk.sort(key=lambda j: (j['input'], j['method'] != 'A', j['K'], j['method']))
        else:
            rng.shuffle(blk)
        out += blk
    if len({j['id'] for j in out}) != len(out): raise SystemExit('duplicate job ids in plan')
    return out


def du_gb(path):
    return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file()) / 2 ** 30


def _load(runp, j):
    folder = runp / 'workers' / j['id']
    return folder, np.load(folder / 'output.npy', mmap_mode='r')


def _drop(folder, arr, keep):
    """Keep a compressed copy only for small BraTS-derived volumes, then free the large .npy (storage cap)."""
    if keep: np.savez_compressed(folder / 'output.npz', u=np.asarray(arr))
    del arr
    (folder / 'output.npy').unlink()


def finalize_sub(runp, a_job, k_jobs):
    """Metadata outputs of ONE (input, K) are present (A is kept for the whole input): compare A / B_K / C_K (contract items 4-7), keep diff coordinates, free B_K and C_K.
    Peak disk for an input = A + B_K + C_K (3 arrays), not all K at once."""
    fields, Ks, paths = {}, {}, {}
    for j in ([a_job] if a_job else []) + k_jobs:
        folder, arr = _load(runp, j)
        lab = 'A' if j['method'] == 'A' else f"{j['method']}{j['K']}"
        fields[lab], paths[lab] = arr, folder
        if j['method'] != 'A': Ks[lab] = j['K']
    vol = np.load(next(iter(paths.values())) / 'vol.npy')
    big = vol.shape[2] > 400                                                                 # copies >= 3 (z > 400): skip component analysis (label arrays would exceed the RAM budget)
    res, coords = CMP.compare_group(vol, {k: np.asarray(v) for k, v in fields.items()}, Ks, thresholds=not big)
    tag = f"{(a_job or k_jobs[0])['input'].replace(':', '-')}_K{k_jobs[0]['K']}"
    C.writej(runp / 'compare' / f'{tag}.json', {'input': (a_job or k_jobs[0])['input'], 'K': k_jobs[0]['K'], 'labels': sorted(fields), 'thresholds_skipped_for_size': big, **res})
    if coords:
        np.savez_compressed(runp / 'compare' / f'{tag}_diffcells.npz', **{k: v for k, v in coords.items()})
    keep = k_jobs[0]['input'].startswith(('brats:', 'bratsinj:')) and vol.shape[2] <= 156
    for lab in [l for l in paths if l != 'A']:
        _drop(paths[lab], fields.pop(lab), keep)
    del vol, fields
    C.seal(runp)
    gb = du_gb(ROOT / 'results')
    if gb > CAP_GB: raise RuntimeError(f'storage cap: results/ is {gb:.2f} GB > {CAP_GB} GB; expansion stopped, separate approval required')


def finalize_input(runp, a_job, keep_small):
    if a_job:
        folder, arr = _load(runp, a_job)
        _drop(folder, arr, keep_small)
        C.seal(runp)


def child(c, cfg, backend, runp, j):
    folder = runp / 'workers' / j['id']
    if folder.exists():
        rec = folder / 'record.json'
        if rec.is_file():
            r = C.readj(rec)
            if r.get('status') == 'COMPLETED' and r.get('validation', {}).get('ok'): return r
        raise RuntimeError('incomplete/failed worker preserved; use a new run directory: ' + str(folder))
    args = [sys.executable, '-B', str(ROOT / 'scripts/r8f1l_worker.py'), '--config', str(cfg), '--out', str(folder), '--input', j['input'], '--method', j['method'],
            '--K', str(j['K']), '--purpose', j['purpose'], '--rep', str(j['rep']), '--stage', j['stage'], '--backend', backend]
    if j.get('cpu_ref'): args.append('--cpu-ref')
    if j['purpose'] == 'timing': args.append('--discard')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', CUDA_VISIBLE_DEVICES=str(c['gpu_index']), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', CUPY_CACHE_DIR=str(runp / 'cupy_cache'))
    (runp / 'logs').mkdir(exist_ok=True)
    with (runp / 'logs' / (j['id'] + '.log')).open('x', encoding='utf-8') as f:
        proc = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, env=env)
        t0 = time.monotonic()
        try:
            while proc.poll() is None:
                if (runp / 'STOP_REQUESTED').exists(): raise RuntimeError('USER_STOP_REQUESTED')
                if time.monotonic() - t0 > c['worker_timeout_s']: raise TimeoutError('worker timeout')
                try:
                    st = C.readj(folder / 'stage.json')                                          # worker replaces this file atomically; a reader can still see it missing/partial for an instant
                except (FileNotFoundError, ValueError):
                    st = None
                if st and time.time() - st['started_unix_s'] > c['stage_timeout_s']: raise TimeoutError('stage timeout: ' + st['stage'])
                time.sleep(.5)
        except BaseException as e:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            C.writej(folder / 'controller_interruption.json', {'reason': str(e), 'job': j})
            raise
    r = C.readj(folder / 'record.json')
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--backend', choices=('cupy', 'stub'), default='cupy')
    ap.add_argument('--plan-file', type=Path, help='TEST ONLY: JSON job list replacing the registered plan')
    ap.add_argument('--dry-plan', action='store_true')
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    runp = a.run.resolve()
    res = (ROOT / 'results').resolve()
    if runp == res or not runp.is_relative_to(res): raise SystemExit('run dir must be a child of results/')
    js = json.loads(a.plan_file.read_text()) if a.plan_file else make_plan(a.stage, runp)
    n = {p: sum(j['purpose'] == p for j in js) for p in ('smoke', 'timing', 'metadata', 'profile')}
    est = EST_H.get(a.stage, 0.01)
    spent = L.ledger_spent()
    g_prior = L.ledger_spent(L.GROUP_OF[a.stage])                                              # prechecks of the same budget group (e.g. S0: b_equiv/negctl/timer) count against the stage's 1.5x rule
    if a.dry_plan:
        print(json.dumps({'stage': a.stage, 'jobs': len(js), 'by_purpose': n, 'est_h': est, 'stop_if_wall_h_gt': est * WALL_RULE, 'sum_all_stage_estimates_h': sum(EST_H.values()),
                          'estimates_within_cap': sum(EST_H.values()) <= CAP_H, 'ledger_spent_h': spent, 'ledger_group_spent_h': g_prior}, indent=1))
        return 0
    if not a.plan_file and g_prior > est * WALL_RULE: raise SystemExit(f'budget: group {L.GROUP_OF[a.stage]} already used {g_prior:.2f} h > {WALL_RULE}x estimate {est} h')
    if spent + max(est - g_prior, 0) > CAP_H: raise SystemExit(f'budget: ledger {spent:.2f} h + stage estimate {est} h > {CAP_H} h cap; separate approval required')
    if a.stage in ('S3', 'S3b', 'S4') and not (runp / 'GATE_S2_PASS.json').is_file():
        raise SystemExit('Gate G-S2 not passed: S3-S4 not allowed (write GATE_S2_PASS.json via r8f1l_gate.py after S0-S2 review)')
    c = D.config(a.config)
    runp.mkdir(parents=True, exist_ok=True)
    if (runp / 'STOPPED.json').exists() or (runp / 'COMPLETED.json').exists(): raise SystemExit('terminal run preserved; use a new directory')
    if any(runp.iterdir()) and not a.resume: raise SystemExit('existing run: use --resume or a new directory')
    t_start = time.monotonic()
    try:
        C.writej(runp / f'plan_{a.stage}.json', {'stage': a.stage, 'jobs': js, 'own_sources': L.own_hashes(), 'config_sha256': C.obj_sha(c), 'est_h': est, 'freeze_manifest_sha256': C.sha(ROOT / 'FREEZE_MANIFEST.json') if (ROOT / 'FREEZE_MANIFEST.json').is_file() else None})
    except FileExistsError: pass
    done, gpu_s = {}, 0.0
    groups = {}
    for j in js:
        if j['purpose'] == 'metadata': groups.setdefault(j['input'], []).append(j)
    finished = {}
    try:
        for i, j in enumerate(js):
            if (gpu_s / 3600 + g_prior) > est * WALL_RULE: raise RuntimeError(f'wall rule: stage {a.stage} exceeded {WALL_RULE}x estimate ({est} h); remaining jobs not started')
            if spent + gpu_s / 3600 > CAP_H: raise RuntimeError(f'budget: cumulative GPU wall exceeded {CAP_H} h cap')
            print(f'[{i + 1}/{len(js)}] {j["id"]}', flush=True)
            t_job = time.monotonic()
            r = child(c, a.config.resolve(), a.backend, runp, j)
            gpu_s += time.monotonic() - t_job
            if r.get('status') == 'ADMISSION_REFUSED':                                       # protocol 6.3: recorded, not a stop
                done[j['id']] = 'NOT_RUNNABLE'; continue
            v = L.validate_record(r, runp / 'workers' / j['id'], allow_stub=a.backend == 'stub', require_regression=bool(j.get('regression')))
            if not v['ok'] or r.get('status') != 'COMPLETED':
                raise RuntimeError(f'DEFECT in {j["id"]}: {r.get("error") or v["reasons"]}')                  # contract 1-3 violation => stop, preserve (protocol 6.1)
            key = (j['input'], j['method'], j['K'])
            done.setdefault(key, set()).add(r['out_sha256'])
            if len(done[key]) != 1: raise RuntimeError(f'DEFECT: nondeterministic output across passes for {key}')
            if j['purpose'] == 'metadata':
                inp = j['input']
                fin = finished.setdefault(inp, [])
                fin.append(j)
                a_job = next((x for x in groups[inp] if x['method'] == 'A'), None)
                if len(groups[inp]) > 1:
                    if j['method'] != 'A':
                        need = [x for x in groups[inp] if x['method'] != 'A' and x['K'] == j['K']]
                        if all(x in fin for x in need) and (a_job is None or a_job in fin):
                            finalize_sub(runp, a_job, need)
                    if len(fin) == len(groups[inp]):
                        finalize_input(runp, a_job, inp.startswith(('brats:', 'bratsinj:')) and r['vol_shape'][2] <= 156)
        ledger_add({'backend': a.backend, 'stage': a.stage, 'run': str(runp), 'gpu_wall_h': gpu_s / 3600, 'wall_h': (time.monotonic() - t_start) / 3600, 'status': 'COMPLETED'})
        C.writej(runp / 'COMPLETED.json', {'stage': a.stage, 'jobs': len(js), 'wall_h': (time.monotonic() - t_start) / 3600, 'gpu_wall_h': gpu_s / 3600})
        C.seal(runp)
        print('COMPLETED', a.stage)
        return 0
    except BaseException as e:
        ledger_add({'backend': a.backend, 'stage': a.stage, 'run': str(runp), 'gpu_wall_h': gpu_s / 3600, 'wall_h': (time.monotonic() - t_start) / 3600, 'status': 'STOPPED'})
        C.writej(runp / 'STOPPED.json', {'reason': f'{type(e).__name__}: {e}', 'action': 'partial evidence preserved; diagnose, then use a new run directory', 'wall_h': (time.monotonic() - t_start) / 3600})
        C.seal(runp)
        print(str(e), file=sys.stderr)
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
