"""One fresh process = one (input, method, K, purpose, rep) measurement.  A = single FP (K=1), B = run_ibi_v10(max_rounds=0), C = run_ibi_v10(max_rounds=8).
New code is limited to: method axis, synthetic/injected inputs, discard run, kernel-event cert timing, contention + transfer records, B-aware validation.
Never edits production or vendored R8-D/R8-F1J files.  allow_shared=True: other GPU users are recorded, never blocked on."""
import argparse
import contextlib
import gc
import os
import time
import traceback
from pathlib import Path

import numpy as np

import r8f1l_lib as L
from r8f1l_lib import C, D, METHODS, ROOT, algorithm, check_array
from r8c_gpu_memory_probe import verify
from r8d_activation_trace import ActivationTrace
from r8d_memory_trace import MemoryTrace
from r8d_timing_probe import Profiler


def build_volume(spec, env, be, c):
    import numpy as np
    if spec['kind'] == 'smoke':
        vol = np.random.default_rng(20261016).integers(0, 256, tuple(c['smoke_shape']), dtype=np.uint8)
        return vol, L.arr_sha(vol)
    if spec['kind'] in ('synth', 'synthinj'):
        vol = L.synth_volume(spec)
        if spec['kind'] == 'synthinj':
            vol = L.inject(vol, spec['struct'], spec['pos'])
        return vol, L.arr_sha(vol)
    import nibabel as nib
    entry = env['inventory'][spec['subject']]
    base = be.preprocess(nib.load(entry['path']).get_fdata(dtype=np.float32))
    if L.arr_sha(base) != entry['vol_sha256']:
        raise RuntimeError('preprocessed hash mismatch')
    vol = np.ascontiguousarray(np.concatenate([base if i % 2 == 0 else base[:, :, ::-1] for i in range(spec['copies'])], axis=2))
    if spec['kind'] == 'bratsinj':
        vol = L.inject(vol, spec['struct'], spec['pos'])
    return vol, entry['sha256']


def cpu_reference(vol, method, K, rounds):
    import cpu_fp, ibi_cpu, seedref
    if method == 'A':
        lo, hi, li = cpu_fp.build_ispan_cpu(vol)
        return seedref.seed_aware_cpu_reference(lo, hi, li, None, 'lower')[0][1:-1, 1:-1, 1:-1]
    return ibi_cpu.ibi(vol, K, 1, rounds, tie='lower')['u']


def worker(a):
    meth = METHODS[a.method]
    K = 1 if a.method == 'A' else a.K
    c = D.config(a.config)
    if meth['max_rounds'] is not None:
        c['max_rounds'] = meth['max_rounds']
    out = Path(a.out).resolve()
    res = (ROOT / 'results').resolve()
    if out == res or not out.is_relative_to(res):
        raise SystemExit('output must be a child of R8F1L/results')
    out.mkdir(parents=True, exist_ok=False)
    spec = L.parse_input(a.input)
    row = {'schema': 'R8F1L-1', 'status': 'STARTED', 'backend': a.backend, 'cuda_executed': False, 'input': a.input, 'stage': a.stage, 'method_id': a.method,
           'method': meth['label'], 'K': K, 'purpose': a.purpose, 'rep': a.rep, 'config_sha256': C.obj_sha(c), 'max_rounds': c['max_rounds'], 'pid': os.getpid(),
           'cpu_ref_requested': bool(a.cpu_ref)}
    C.writej(out / 'partial.json', row)
    stage = lambda n: C.writej(out / 'stage.json', {'stage': n, 'started_unix_s': time.time()})
    prof = None
    try:
        idle_pre = L.smi_snapshot(os.getpid()) if a.backend == 'cupy' else {'util': None}
        stage('environment')
        env = D.gate(c, require_data=spec['kind'] in ('brats', 'bratsinj')) if a.backend == 'cupy' else {'ok': True, 'stub': True, 'sources': {}, 'inventory': {}}
        env['own_sources'] = L.own_hashes()
        C.writej(out / 'environment.json', env)
        if not env['ok']:
            raise RuntimeError('; '.join(env['reasons']))
        R7 = D.load_r7(c)
        lib = str(Path(c['r7_runner']) / 'lib')
        be = R7.CupyBackend(c['repo'], lib) if a.backend == 'cupy' else R7.StubBackend(c['repo'], lib, None, K)
        cp = be.cp if be.cuda else None
        stage('input')
        vol, input_hash = build_volume(spec, env, be, c)
        if K > 1 and vol.shape[2] // K < 2:
            raise RuntimeError('slab thickness < 2 original planes')
        stage('admission')
        adm = C.admit(c, tuple(int(v) for v in vol.shape), K) if be.cuda else {'ok': True, 'vram_limit_mib': 1e9}
        C.writej(out / 'admission.json', adm)
        if not adm['ok']:
            raise RuntimeError('ADMISSION REFUSED: ' + '; '.join(adm['reasons']))
        stage('warmup')
        t = time.perf_counter()
        warm = np.random.default_rng(1).integers(0, 256, tuple(c['smoke_shape']), dtype=np.uint8)
        wu, _ = algorithm(be, warm, a.method, min(K, c['smoke_shape'][2] // 2) if K > 1 else 1, c['max_rounds'])
        del wu, warm
        if be.cuda:
            tiny = cp.zeros((9, 9, 9), dtype=cp.float32)
            be.global_check(tiny)
            del tiny
        be.sync()
        row['warmup_wall_s'] = time.perf_counter() - t
        gc.collect()
        be.free()
        if be.cuda:
            cp.get_default_pinned_memory_pool().free_all_blocks()
            adm = C.admit(c, tuple(int(v) for v in vol.shape), K)
            C.writej(out / 'admission.json', adm)
            if not adm['ok']:
                raise RuntimeError('resources changed during warmup')
            cp.get_default_memory_pool().set_limit(size=int(adm['vram_limit_mib'] * 2 ** 20))
        be.sync()
        if a.purpose == 'timing' and a.discard:           # full-size discarded run: pool/JIT steady state (timing passes only)
            stage('discard')
            t = time.perf_counter()
            u0, _ = algorithm(be, vol, a.method, K, c['max_rounds'])
            row['discard_out_sha256'] = L.arr_sha(u0)
            del u0
            be.sync()
            row['discard_wall_s'] = time.perf_counter() - t
            gc.collect()
        if a.purpose == 'profile':
            prof = Profiler(cp, out / 'profile.jsonl', c['stage_timeout_s'])
        tracing = ActivationTrace(be, prof) if a.purpose != 'timing' and be.cuda else contextlib.nullcontext()
        monitoring = MemoryTrace(cp) if a.purpose == 'metadata' and be.cuda else contextlib.nullcontext()
        tally = L.TransferTally(cp) if a.purpose == 'metadata' and be.cuda else contextlib.nullcontext()
        mon = L.ContentionMonitor(os.getpid()) if be.cuda else contextlib.nullcontext()
        stage('algorithm')
        with mon, monitoring as memory, tally as tt, tracing as trace:
            be.sync()
            if be.cuda:
                e0, e1 = cp.cuda.Event(), cp.cuda.Event()
                e0.record()
            t = time.perf_counter()
            with prof.span('algorithm_profile') if prof else contextlib.nullcontext():
                u, extra = algorithm(be, vol, a.method, K, c['max_rounds'], prof)
            be.sync()
            row['algorithm_wall_s'] = time.perf_counter() - t
            if be.cuda:
                e1.record()
                e1.synchronize()
                row['cuda_event_envelope_s'] = cp.cuda.get_elapsed_time(e0, e1) / 1000
            else:
                row['cuda_event_envelope_s'] = row['algorithm_wall_s']
        if prof:
            row['profile'] = prof.result()
            prof.close()
            prof = None
        if trace:
            row['activation'] = trace.result()
        if memory:
            row['memory'] = memory.result()
        if tt:
            row['transfers'] = tt.result()
        row.update(extra, cuda_executed=be.cuda, vol_shape=list(vol.shape), out_shape=list(u.shape), input_sha256=input_hash, vol_sha256=L.arr_sha(vol), out_sha256=L.arr_sha(u))
        stage('verification')
        row['verification'] = verify(be, u, extra)
        row['array_check'] = check_array(u, vol)
        if be.cuda:
            row['cert_kernel'] = L.cert_kernel_event(be, u)
            be.sync()
            time.sleep(1.2)                                # idle window: any residual util is NOT ours
            idle_post = L.smi_snapshot(os.getpid())
            row['gpu_state'] = mon.result(idle_pre, idle_post)
        else:
            row['gpu_state'] = {'contended': False, 'idle_pre': {'util': None}, 'smi_unavailable': True}
        v = row['verification']
        row['t_total_s'] = {'alg': row['algorithm_wall_s'], 'cert_wall': v['t_global_s'], 'h2d': v['t_h2d_s'], 'direct': row['algorithm_wall_s'] + v['t_global_s'],
                            'direct_with_h2d': row['algorithm_wall_s'] + v['t_global_s'] + v['t_h2d_s']}
        if a.cpu_ref:
            stage('cpu_reference')
            ref = cpu_reference(vol, a.method, K, c['max_rounds'])
            row['cpu_reference_equal'] = bool(np.array_equal(np.ascontiguousarray(u).view(np.uint32), np.ascontiguousarray(ref, np.float32).view(np.uint32)))
            row['cpu_dwc_violations'] = int(sum(be.cpu_check(u).values()))
            bad = np.zeros((3, 3, 3), np.float32)
            bad[0, 0, 0] = bad[1, 1, 1] = 1
            row['negative_control_detected'] = verify(be, bad, {})['global_violations'] > 0 and sum(be.cpu_check(bad).values()) > 0
        row['regression'] = {'status': 'NOT AVAILABLE'}
        if spec['kind'] == 'brats' and spec['copies'] == 1 and a.method in ('A', 'C') and (a.method == 'A' or K == 16) and be.cuda:
            e1j = C.readj(Path(c['e1_subjects']) / (spec['subject'] + '.json'))
            exp = e1j['hashes']['gpu_k1_out'] if a.method == 'A' else e1j['ddfp']['out_sha256']
            row['regression'] = {'expected': exp, 'ok': row['out_sha256'] == exp}
        stage('save')
        if a.purpose == 'metadata':
            np.save(out / 'output.npy', u, allow_pickle=False)
            row['output_file_sha256'] = C.sha(out / 'output.npy')
            np.save(out / 'vol.npy', vol, allow_pickle=False)           # uint8 input (<= 36 MB) for the fidelity comparison
        row['environment_sha256'], row['admission_sha256'] = C.sha(out / 'environment.json'), C.sha(out / 'admission.json')
        row['sources_unchanged'] = (D.hashes(c) == env['sources']) if be.cuda else True
        row['own_sources_unchanged'] = L.own_hashes() == env['own_sources']
        row['status'] = 'COMPLETED'
        row['validation'] = L.validate_record(row, out, allow_stub=a.backend == 'stub')
        if not row['validation']['ok']:
            row['status'] = 'VALIDATION_FAILED'
        C.writej(out / 'record.json', row)
        stage('done')
        C.seal(out)
        return 0 if row['validation']['ok'] else 3
    except BaseException as e:
        row.update(status='INTERRUPTED' if isinstance(e, KeyboardInterrupt) else 'ADMISSION_REFUSED' if 'ADMISSION' in str(e) else 'OOM' if 'out of memory' in str(e).lower() else 'ERROR',
                   error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())
        if prof:
            prof.close()
        C.writej(out / 'record.json', row)
        C.seal(out)
        print(row['error'], flush=True)
        return 3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('config', 'out', 'input', 'purpose', 'stage'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--method', choices=sorted(METHODS), required=True)
    p.add_argument('--backend', choices=('cupy', 'stub'), default='cupy')
    for n in ('K', 'rep'):
        p.add_argument('--' + n, type=int, required=True)
    p.add_argument('--cpu-ref', action='store_true')
    p.add_argument('--discard', action='store_true')
    a = p.parse_args()
    a.cpu_ref = a.cpu_ref or a.purpose == 'smoke'
    if a.purpose not in ('smoke', 'timing', 'profile', 'metadata') or (a.method != 'A' and a.K < 2) or (a.method == 'A' and a.K != 1):
        p.error('invalid purpose/K/method combination')
    return worker(a)


if __name__ == '__main__':
    raise SystemExit(main())
