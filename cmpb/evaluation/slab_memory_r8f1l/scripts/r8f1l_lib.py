"""R8-F1L shared helpers (NEW code; frozen R8-D / R8-F1J sources are imported from vendor/, never edited).

Design (ponytail): reuse R8-D `config/gate/admit/verify/check_array/MemoryTrace/Profiler/ActivationTrace`; add only
  (1) a B-capable method axis (A | B | C), (2) synthetic inputs, (3) kernel-event + contention + transfer measurement,
  (4) a fail-closed record validator that accepts B (R_star == 0), which the frozen R8-D validator cannot.
Everything GPU-specific is guarded by `be.cuda`; the stub backend (CPU stand-ins) exists only to test plumbing.
"""
import hashlib
import json
import math
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'vendor/r8d/scripts'), str(ROOT / 'vendor/f1j')]
import r8d_common as D                      # noqa: E402  (also puts vendor/r8c/scripts on sys.path)
import r8f1j_oracle as O                    # noqa: E402
from r8c_memory_model import extent, verifier_plan   # noqa: E402
from r8c_validate import check_array        # noqa: E402

C = D.C
SEED = 20271018

# ----------------------------------------------------------------------------- GPU-hour budget (single ledger for runner AND prechecks)
EST_H = {'S0': 0.5, 'S1': 0.5, 'S2': 1.7, 'S3': 2.7, 'S3b': 1.4, 'S4': 0.8}      # frozen BEFORE any GPU run; S3b = worst case (union of two K ladders, 104 jobs); sum 7.6 h
CAP_H, WALL_RULE, CAP_GB = 9.1, 1.5, 8.0
GROUP_OF = {'PRE_b_equiv': 'S0', 'PRE_negctl': 'S0', 'PRE_timer': 'S0', 'S0': 'S0', 'PRE_s1': 'S1', 'S2': 'S2', 'S3': 'S3', 'S3b': 'S3b', 'S4': 'S4'}   # S0 budget covers its prechecks (F1K S0-3..S0-5)


def ledger_path():
    return ROOT / 'results/LEDGER.json'


def ledger_read(path=None):
    p = Path(path) if path else ledger_path()
    return C.readj(p) if p.is_file() else []


def ledger_add(entry, path=None, record_stub=False):
    """Append one entry {stage, gpu_wall_h, ...}; stub (plumbing) runs never consume GPU hours unless a test asks for it."""
    if entry.get('backend') == 'stub' and not record_stub: return
    assert entry['stage'] in GROUP_OF and entry['gpu_wall_h'] >= 0
    p = Path(path) if path else ledger_path()
    C.writej(p, ledger_read(p) + [dict(entry, group=GROUP_OF[entry['stage']])])


def ledger_spent(group=None, path=None):
    return sum(e['gpu_wall_h'] for e in ledger_read(path) if group is None or e.get('group') == group)


def budget_guard(stage, path=None):
    """Raise before any GPU work when the cap or the stage's 1.5x wall rule is already exhausted. Returns hours already spent in the stage's group."""
    g = GROUP_OF[stage]
    if ledger_spent(None, path) > CAP_H: raise SystemExit(f'budget: ledger {ledger_spent(None, path):.2f} h > {CAP_H} h cap; separate approval required')
    if ledger_spent(g, path) > EST_H[g] * WALL_RULE: raise SystemExit(f'budget: group {g} already used {ledger_spent(g, path):.2f} h > {WALL_RULE}x estimate {EST_H[g]} h')
    return ledger_spent(g, path)
METHODS = {'A': {'label': 'gpu_k1', 'max_rounds': None}, 'B': {'label': 'B_round0', 'max_rounds': 0}, 'C': {'label': 'C_ibi', 'max_rounds': 8}}
HEX64 = re.compile('[a-f0-9]{64}')


# ----------------------------------------------------------------------------- inputs
def parse_input(spec):
    """smoke | brats:<subject>:<copies> | synth:<family>:<WxHxD>:<alphabet>:<seed>"""
    p = spec.split(':')
    if p == ['smoke']:
        return {'kind': 'smoke'}
    if p[0] == 'brats' and len(p) == 3 and p[2] in ('1', '2', '4', '6', '8'):
        return {'kind': 'brats', 'subject': p[1], 'copies': int(p[2])}
    if p[0] == 'synth' and len(p) == 5 and p[1] in O.FAMILIES:
        shape = tuple(int(x) for x in p[2].split('x'))
        if len(shape) != 3 or min(shape) < 1 or not 2 <= int(p[3]) <= 256:
            raise ValueError('bad synth shape/alphabet: ' + spec)
        return {'kind': 'synth', 'family': p[1], 'shape': shape, 'alphabet': int(p[3]), 'seed': int(p[4])}
    if p[0] == 'bratsinj' and len(p) == 4 and p[2] in INJ_KINDS and p[3] in INJ_POS:
        return {'kind': 'bratsinj', 'subject': p[1], 'copies': 1, 'struct': p[2], 'pos': p[3]}
    if p[0] == 'synthinj' and len(p) == 5 and p[1] in INJ_KINDS and p[2] in INJ_POS:
        return {'kind': 'synthinj', 'struct': p[1], 'pos': p[2], 'shape': tuple(int(x) for x in p[3].split('x')), 'seed': int(p[4]), 'family': 'zlayer_noise', 'alphabet': 256}
    raise ValueError('unrecognised input spec: ' + spec)


INJ_KINDS, INJ_POS, INJ_K = ('diag', 'checker'), ('on', 'near', 'far'), 16


def inject(vol, kind, pos, K=INJ_K, half=1):
    """S4 (PROPOSED definition, needs user confirmation): thin structure of 2 original z-planes at the middle seam z_b of the K-slab split.
    on: straddles z_b; near: first structure plane at z_b+1 (expanded distance d=2, band d<=4) without touching plane z_b; far: slab middle (d > 4). Deterministic, uint8."""
    v = vol.copy()
    zb = [z1 for _, z1 in O.split_domain_1d(v.shape[2], K)[:-1]]
    s = len(zb) // 2
    zc = {'on': zb[s], 'near': zb[s] + (3 if kind == 'diag' else 2), 'far': (zb[s] + zb[s + 1]) // 2}[pos]
    x0, y0 = max(0, v.shape[0] // 2 - 24), max(0, v.shape[1] // 2 - 24)
    xs, ys = np.arange(x0, min(v.shape[0], x0 + 48)), np.arange(y0, min(v.shape[1], y0 + 48))
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    for z in range(max(0, zc - half - 3), min(v.shape[2], zc + half + 3)):
        if kind == 'diag':
            m = np.abs((z - zc) - ((X - x0) / 16.0 - 1.5)) <= half
            v[x0:x0 + len(xs), y0:y0 + len(ys), z][m] = 255
        elif zc - half <= z < zc + half:
            v[x0:x0 + len(xs), y0:y0 + len(ys), z] = (((X + Y + z) % 2) * 255).astype(np.uint8)
    return v


def synth_volume(s):
    rng = np.random.default_rng([SEED, s['seed'], O.FAMILIES.index(s['family']), s['alphabet']])
    return np.ascontiguousarray(O.gen_volume(s['family'], s['shape'], s['alphabet'], rng))


def arr_sha(a):
    return D.arr_sha(a)


def own_hashes():
    return {str(p): C.sha(p) for p in sorted((ROOT / 'scripts').glob('*.py'))}


# ----------------------------------------------------------------------------- algorithm window (A | B | C)
def algorithm(be, vol, method, K, rounds, profile=None):
    """Same boundary/host-return behaviour as R8-D `algorithm`: host uint8 volume in, host float32 field out."""
    if method == 'A':
        lo, hi, li = be.prep(vol)
        if be.cuda:
            cp = profile.cp_proxy if profile else be.cp
            u = be._fp(lo, hi, li, verbose=False)
            return cp.asnumpy(u[1:-1, 1:-1, 1:-1]), {}
        return np.ascontiguousarray(be.fp_k1(lo, hi, li)[1:-1, 1:-1, 1:-1]), {}
    r = be.ibi(vol, K, 1, rounds)
    return r['u_dd'], {'R_star': r['R_star'], 'boundary_z_orig': r['boundary_z_orig']}


# ----------------------------------------------------------------------------- measurement helpers
def smi_snapshot(own_pid=None):
    """Read-only nvidia-smi snapshot (device util/mem + compute apps). Never raises; never signals anyone."""
    def q(args):
        try:
            return subprocess.run(['nvidia-smi'] + args + ['--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as e:
            return f'UNAVAILABLE({type(e).__name__})'
    g, a = q(['--query-gpu=utilization.gpu,memory.used,memory.total']), q(['--query-compute-apps=pid,used_memory'])
    out = {'t': time.time(), 'raw_gpu': g, 'raw_apps': a}
    try:
        u, m, t = (float(x) for x in g.splitlines()[0].split(','))
        out.update(util=u, mem_used_mib=m, mem_total_mib=t)
    except Exception:
        out.update(util=None, mem_used_mib=None, mem_total_mib=None)
    apps = {}
    for line in a.splitlines():
        if ',' in line:
            try:
                pid, mib = (int(x) for x in line.split(','))
                apps[pid] = mib
            except ValueError:
                pass
    out['apps'] = {p: m for p, m in apps.items() if p != own_pid}
    return out


class ContentionMonitor:
    """1 Hz sampling of OTHER compute processes during the measured window (allow_shared=True: never blocks, only labels).
    contended := idle device util > 30 % before/after (own CUDA work excluded by timing), or any other-process memory delta > 256 MiB,
    or the set of other compute processes changed. Device-level util is the proxy for 'other-process utilisation' (nvidia-smi has no
    per-process util without pmon). Processes of other containers may be invisible: absence of evidence is recorded, not asserted."""
    def __init__(self, own_pid, period=1.0):
        self.pid, self.period, self.samples, self.stop = own_pid, period, [], threading.Event()
        self.th = None

    def _loop(self):
        while not self.stop.wait(self.period):
            self.samples.append(smi_snapshot(self.pid))

    def __enter__(self):
        self.samples.append(smi_snapshot(self.pid))
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *e):
        self.stop.set()
        self.th.join(5)
        self.samples.append(smi_snapshot(self.pid))
        return False

    def result(self, idle_pre, idle_post):
        sets = [frozenset(s['apps']) for s in self.samples]
        mems = [sum(s['apps'].values()) for s in self.samples]
        delta = (max(mems) - min(mems)) if mems else 0
        utils = [u for u in (idle_pre.get('util'), idle_post.get('util')) if u is not None]
        why = []
        if any(u > 30 for u in utils): why.append('idle_util_gt_30pct')
        if delta > 256: why.append('other_mem_delta_gt_256MiB')
        if len(set(sets)) > 1: why.append('other_process_set_changed')
        unavailable = all(s['util'] is None for s in self.samples)
        return {'n_samples': len(self.samples), 'idle_pre': idle_pre, 'idle_post': idle_post, 'other_apps_seen': sorted({p for s in sets for p in s}),
                'other_mem_delta_mib': delta, 'contended': bool(why), 'contended_reasons': why, 'smi_unavailable': unavailable,
                'note': 'sampler is a 1 Hz subprocess poll; its host overhead is part of every measured pass'}


class TransferTally:
    """Counts cross-device cupy API calls (asnumpy D2H, asarray/array H2D) on the REAL module during a NON-timed pass.
    Implicit transfers (ndarray.item/get/__getitem__ scalars) and driver staging are not observed."""
    def __init__(self, cp):
        self.cp, self.t, self.old = cp, {}, {}

    def __enter__(self):
        cp = self.cp
        for name in ('asnumpy', 'asarray', 'array'):
            fn = getattr(cp, name)
            self.old[name] = fn

            def wrap(*a, _n=name, _f=fn, **k):
                cross = bool(a) and ((_n == 'asnumpy' and isinstance(a[0], cp.ndarray)) or (_n != 'asnumpy' and not isinstance(a[0], cp.ndarray)))
                if not cross:
                    return _f(*a, **k)
                t0 = time.perf_counter()
                r = _f(*a, **k)
                d = 'D2H' if _n == 'asnumpy' else 'H2D'
                b = self.t.setdefault(d, {'calls': 0, 'bytes': 0, 'host_api_s': 0.0})
                b['calls'] += 1
                b['bytes'] += int(getattr(r, 'nbytes', 0))
                b['host_api_s'] += time.perf_counter() - t0
                return r
            setattr(cp, name, wrap)
        return self

    def __exit__(self, *e):
        for n, f in self.old.items():
            setattr(self.cp, n, f)
        return False

    def result(self):
        return {'transfers': self.t, 'scope': 'cupy.asnumpy/asarray/array API payloads only; excludes ndarray.get/.item/index scalars and staging'}


def cert_kernel_event(be, u):
    """Event-bracketed launch of the PRODUCTION dwc_check_all_blocks kernel alone (same grid/block as verify_dwc_gpu).
    Cross-checks counts against verify_dwc_gpu. Includes neither H2D nor the float32 copy."""
    cp = be.cp
    import src.ddfp.gpu_immersion as gi
    k = gi._get_nan_kernels()[1]
    ug = be.h2d(u)
    W, H, Dp = ug.shape
    flat = ug.ravel().astype(cp.float32)
    total = (W-1)*(H-1)*Dp + (W-1)*H*(Dp-1) + W*(H-1)*(Dp-1) + (W-1)*(H-1)*(Dp-1)
    stats = cp.zeros(2, dtype=cp.int32)
    be.sync()
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record()
    k(((total + 255) // 256,), (256,), (flat, stats, W, H, Dp))
    e1.record()
    e1.synchronize()
    out = {'kernel_event_s': cp.cuda.get_elapsed_time(e0, e1) / 1000, 'violations': int(stats[0]), 'nan_skipped': int(stats[1]), 'total_blocks': int(total),
           'scope': 'single kernel launch on null stream; excludes H2D, float32 copy and host sync cost'}
    del ug, flat, stats
    return out


# ----------------------------------------------------------------------------- validation (fail-closed; accepts B)
def validate_record(r, folder, allow_stub=False, require_regression=False):
    """Returns {'ok', 'reasons'}. Mirrors R8-D contract items 1-3 + provenance, but method-aware (A | B | C)."""
    why = []
    m, K, purpose = r.get('method_id'), r.get('K'), r.get('purpose')
    if r.get('schema') != 'R8F1L-1' or r.get('status') != 'COMPLETED': why.append('not a completed R8F1L-1 record')
    if not (r.get('backend') == 'cupy' and r.get('cuda_executed') is True) and not (allow_stub and r.get('backend') == 'stub'):
        why.append('not real CUDA')
    if m not in METHODS: why.append('unknown method'); return {'ok': False, 'reasons': why}
    want_rounds = METHODS[m]['max_rounds']
    if m == 'A':
        if K != 1: why.append('A must have K=1')
    else:
        if type(K) is not int or K < 2: why.append('B/C need K>=2')
        if r.get('max_rounds') != want_rounds: why.append(f'max_rounds must be {want_rounds}')
    for k in ('out_sha256', 'input_sha256', 'vol_sha256', 'config_sha256'):
        if not HEX64.fullmatch(str(r.get(k, ''))): why.append('invalid ' + k)
    if r.get('sources_unchanged') is not True or r.get('own_sources_unchanged') is not True: why.append('source preservation missing')
    try:
        es = extent(tuple(r['vol_shape']))
        if list(es) != r.get('out_shape') or not verifier_plan(es)['ok']: why.append('shape/int32 unsafe')
    except Exception:
        why.append('invalid shape')
    if r.get('array_check', {}).get('ok') is not True: why.append('items 1-2 (original voxels / ISpan interval / finite) unconfirmed')
    v = r.get('verification', {})
    if not (v.get('verify_status') == 'DONE' and v.get('global_dwc_ok') is True and v.get('global_violations') == 0 and v.get('nan_skipped') == 0):
        why.append('item 3 (global face/cube DWC with nan_skipped==0) missing/failed')
    if m != 'A' and v.get('seam_violations') != 0: why.append('seam check missing/failed')
    if m == 'B' and r.get('R_star') != 0: why.append('B must report R_star == 0')
    if m == 'C' and not (type(r.get('R_star')) is int and 1 <= r['R_star'] < 8): why.append('C convergence strictly before cap unconfirmed')
    for k in ('algorithm_wall_s', 'warmup_wall_s', 'cuda_event_envelope_s'):
        if type(r.get(k)) not in (int, float) or not math.isfinite(r[k]) or r[k] <= 0: why.append('invalid time ' + k)
    if r.get('backend') == 'cupy':
        ck = r.get('cert_kernel', {})
        if not (ck.get('violations') == v.get('global_violations') and ck.get('nan_skipped') == v.get('nan_skipped') and ck.get('total_blocks') == v.get('plan', {}).get('quantities', {}).get('total_blocks')):
            why.append('kernel-event certification counts disagree with verify_dwc_gpu')
        if type(ck.get('kernel_event_s')) not in (int, float) or not ck.get('kernel_event_s', 0) > 0: why.append('cert kernel time missing')
    gs = r.get('gpu_state', {})
    if 'contended' not in gs or 'idle_pre' not in gs: why.append('gpu_state/contention record missing')
    stub = allow_stub and r.get('backend') == 'stub'          # stub = CPU stand-ins: activation/memory/profile instrumentation does not exist there
    if purpose in ('profile', 'metadata') and not stub:
        tr = r.get('activation', {})
        solves = tr.get('solves', [])
        exp = 1 if m == 'A' else K * (1 + (r.get('R_star') or 0))
        if tr.get('n_solves') != exp or len(solves) != exp: why.append('solve coverage mismatch')
        if any(s.get('nan_count') != 0 for s in solves): why.append('FP NaN')
        seeded = sum(bool(s.get('seeded')) for s in solves)
        if m == 'B' and (seeded != 0 or tr.get('round_changes')): why.append('B executed seeded/IBI work')
        if m == 'C' and (seeded != K * r.get('R_star', 0) or [q.get('round') for q in tr.get('round_changes', [])] != list(range(1, r.get('R_star', 0) + 1))
                         or not 0 <= tr['round_changes'][-1].get('max_change', -1) < .5):
            why.append('C round/seed accounting or stop predicate failed')
        if m == 'A' and seeded: why.append('A executed seeded work')
        if tr.get('R_act_total') != sum(s.get('R_act', 0) for s in solves): why.append('activation aggregate mismatch')
    if purpose == 'metadata':
        mem = r.get('memory', {})
        if not stub:
            if mem.get('errors') != [] or mem.get('samples', 0) < 1: why.append('memory monitor failed')
            for k in ('pool_live_peak_bytes', 'pool_reserved_peak_bytes', 'sampled_host_rss_peak_bytes'):
                if type(mem.get(k)) not in (int, float) or not math.isfinite(mem[k]) or mem[k] <= 0: why.append('missing memory ' + k)
            if mem.get('pool_reserved_peak_bytes', 0) < mem.get('pool_live_peak_bytes', 0): why.append('reserved < live')
        p = Path(folder) / 'output.npy'
        if not p.is_file() or C.sha(p) != r.get('output_file_sha256'): why.append('saved output missing/hash')
        else:
            u = np.load(p, mmap_mode='r', allow_pickle=False)
            if list(u.shape) != r.get('out_shape') or u.dtype != np.float32 or arr_sha(u) != r.get('out_sha256'): why.append('array payload hash/shape/dtype')
    if purpose == 'profile' and not stub:
        pr = r.get('profile', {})
        sc = pr.get('nested_inclusive_spans', {})
        if pr.get('profile_intrusive') is not True or not sc.get('build_ispan_gpu', {}).get('calls'): why.append('profile missing')
        tr = r.get('activation', {})
        for lab in ('kernel:activate_level', 'kernel:expand_frontier'):
            if sc.get(lab, {}).get('calls') != tr.get('R_act_total'): why.append(lab + ' activation-count mismatch')
        if sc.get('fp_api', {}).get('calls') != tr.get('n_solves'): why.append('FP profile coverage mismatch')
        if any(s.get('original_slab_constraints', {}).get('ok') is not True for s in tr.get('solves', [])): why.append('per-FP original voxel/interval contract failed')
    reg = r.get('regression', {})
    if reg.get('expected') and reg.get('ok') is not True: why.append('known-output regression mismatch')
    if require_regression and not (HEX64.fullmatch(str(reg.get('expected', ''))) and reg.get('expected') == r.get('out_sha256')): why.append('mandatory known-output regression missing')
    if r.get('cpu_ref_requested'):
        if not (r.get('cpu_reference_equal') is True and r.get('cpu_dwc_violations') == 0 and r.get('negative_control_detected') is True): why.append('CPU reference/negative control failed')
    if r.get('purpose') == 'timing' and r.get('discard_out_sha256') not in (None, r.get('out_sha256')): why.append('discard-run output differs from measured run (nondeterminism)')
    for fn, key in (('environment.json', 'environment_sha256'), ('admission.json', 'admission_sha256')):
        p = Path(folder) / fn
        if not p.is_file() or C.sha(p) != r.get(key): why.append('missing/failing provenance ' + fn)
        elif not allow_stub and C.readj(p).get('ok') is not True: why.append('gate refused ' + fn)
    return {'ok': not why, 'reasons': why}
