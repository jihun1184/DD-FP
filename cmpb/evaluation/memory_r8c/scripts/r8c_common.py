"""Shared fail-closed utilities. Standard library only until a real GPU worker starts."""
import ast
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXPECTED = {
 'scripts/ddfp/experiment_DDFP_all.py': 'b6a9b109f4105e7b77ebed880f669ba670bbd87123deca9509ba5b00993e561e',
 'src/ddfp/cpu_fp.py': '8eae554f7d3d688dee2285f41d20bfbf6e34af98016a3f1ad07f1ff5f8f36a88',
 'src/ddfp/gpu_immersion.py': 'b784a376d3466a1d4ded5089a2eee9f27df35006b4926301dd06106b4ed262a4',
 'src/ddfp/gpu_native_worklist.py': '853bf33a6bbd9c91f5304a018f0a51ff53e8418360fe1477208cf81df17163ac',
 'src/ddfp/gpu_seed_aware_worklist.py': 'c80f56b9ba7afd63196332d51894168d36f6e9f3e2a51b6b966786311e4eecad',
 'test/test_smoke.py': 'd1dea404932f1b022830a5c21cb15daae67e480d5f2d54a882d2b98ac7050811'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def obj_sha(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()
def readj(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def jsonl(path): return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def writej(path, obj, exclusive=False):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if exclusive:
        with p.open('x', encoding='utf-8') as f: f.write(text)
    else:
        tmp = p.with_suffix(p.suffix+'.tmp'); tmp.write_text(text, encoding='utf-8'); os.replace(tmp, p)


def load_config(path):
    c = readj(path)
    for k in ('repo', 'r7_runner', 'brats_root', 'metadata', 'e6_rows', 'e1_subjects', 'python'):
        if not isinstance(c.get(k), str) or not c[k]: raise ValueError(f'missing config {k}')
        c[k] = str(Path(c[k]).expanduser().resolve())
    if c.get('copies') != [1, 2, 4] or c.get('reps') != 3 or c.get('delta') != 1 or c.get('max_rounds') != 8:
        raise ValueError('V1 preregistered primary design is copies=[1,2,4], reps=3, delta=1, max_rounds=8')
    for key in ('reserve_factor', 'reserve_fixed_mib', 'headroom_mib', 'vram_fraction', 'host_fraction', 'worker_timeout_s'):
        v = c.get(key)
        if type(v) not in (float, int) or not math.isfinite(v) or v <= 0: raise ValueError(f'invalid {key}')
    if c['reserve_factor'] < 1 or not 0 < c['vram_fraction'] < 1 or not 0 < c['host_fraction'] < 1:
        raise ValueError('invalid safety margins')
    if type(c.get('gpu_index')) is not int or c['gpu_index'] < 0: raise ValueError('gpu_index must be nonnegative integer')
    return c


def output_path(path, c=None):
    p = Path(path).expanduser().resolve()
    allowed = (HERE / 'results').resolve()
    if p == allowed or not p.is_relative_to(allowed): raise ValueError(f'output must be a child directory of {allowed}')
    if c:
        for k in ('repo', 'r7_runner', 'brats_root', 'e1_subjects'):
            protected = Path(c[k]).resolve()
            if p == protected or p.is_relative_to(protected): raise ValueError(f'protected output path: {p}')
    return p


def source_hashes(c):
    hashes = {}
    for rel, expected in EXPECTED.items():
        p = Path(c['repo']) / rel
        if not p.is_file() or sha(p) != expected: raise ValueError(f'production hash mismatch/missing: {rel}')
        hashes[str(p)] = expected
    for n in ('r7_common.py', 'r7_e4.py'):
        p = Path(c['r7_runner']) / 'r7' / n
        if not p.is_file(): raise ValueError(f'missing R7 helper {p}')
        hashes[str(p)] = sha(p)
    trusted_helpers={'lib/cpu_fp.py':EXPECTED['src/ddfp/cpu_fp.py'],
       'lib/seedref.py':'176e0a59f3330eaec7a32cc073f8269a6b0f53611b64686044aa08da00753296',
       'lib/ibi_cpu.py':'6c361c49267abb468ae54ec95ab5990e7d8f58a1ac8279748991768fe3f351cf',
       'lib/r1_check53.py':'3b4ce6084392f4406e76ae1f15a210fb9850cb74d83f26522d8c0f9bbe7b7571'}
    for rel,expected in trusted_helpers.items():
        p=Path(c['r7_runner'])/rel
        if not p.is_file() or sha(p)!=expected: raise ValueError('frozen reference hash mismatch: '+str(p))
        hashes[str(p)]=expected
    meta = readj(Path(c['e6_rows']).parent/'run_meta.json')
    if hashes[str(Path(c['r7_runner'])/'r7/r7_common.py')] != meta['common_sha256']:
        raise ValueError('R7 common differs from E6 recorded source')
    for p in (HERE/'scripts').glob('*.py'): hashes[str(p)] = sha(p)
    for p in [Path(c['e6_rows']),Path(c['e6_rows']).parent/'run_meta.json',Path(c['metadata'])]+[Path(c['e1_subjects'])/(s+'.json') for s in selection(c)]:
        if not p.is_file(): raise ValueError('missing frozen evidence: '+str(p))
        hashes[str(p)]=sha(p)
    return hashes


def selection(c):
    import statistics as st
    rr = jsonl(c['e6_rows']); groups = {}
    for r in rr:
        if r.get('method') == 'gpu_k1' and r.get('rep', -1) >= 0:
            if r.get('backend') != 'cupy' or r.get('cuda_executed') is not True: raise ValueError('non-CUDA E6 input')
            groups.setdefault(r['subject'], []).append(r)
    if len(groups) != 20 or any(sorted(r['rep'] for r in g) != [0,1,2] for g in groups.values()):
        raise ValueError('E6 selection needs exactly 20 subjects x 3 K1 repetitions')
    rank = sorted(groups, key=lambda s: (st.median(r['t_algorithm_s'] for r in groups[s]), s))
    return [rank[0], rank[9], rank[-1]]


def input_inventory(c):
    metadata = readj(c['metadata']); selected = selection(c); inventory = {}
    for sid in selected:
        rec = next(r for r in metadata['cohort'] if r['subject_id'] == sid)
        p = Path(c['brats_root']) / sid / rec['file_name']
        if not p.is_file() or p.stat().st_size != rec['size_bytes'] or sha(p) != rec['sha256']:
            raise ValueError(f'input missing or SHA256 mismatch: {p}')
        e1 = readj(Path(c['e1_subjects']) / (sid+'.json'))
        if e1['input_sha256'] != rec['sha256']: raise ValueError('E1/input mismatch')
        inventory[sid] = {'path': str(p), 'sha256': rec['sha256'], 'vol_shape': e1['vol_shape'], 'vol_sha256': e1['vol_sha256']}
    return inventory


def smi(c):
    def call(query):
        p = subprocess.run(['nvidia-smi', '-i', str(c['gpu_index']), '--query-'+query, '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        if p.returncode: raise RuntimeError('nvidia-smi failed: '+p.stderr.strip())
        return p.stdout.strip()
    gpu = [s.strip() for s in call('gpu=uuid,name,memory.total,memory.used,utilization.gpu').split(',')]
    if len(gpu) != 5: raise RuntimeError('ambiguous GPU inventory')
    processes = []
    for l in call('compute-apps=gpu_uuid,pid,used_gpu_memory').splitlines():
        if not l.strip(): continue
        uuid, pid, mem = [v.strip() for v in l.split(',')]
        if uuid == gpu[0]: processes.append({'pid': int(pid), 'memory_mib': float(mem), 'gpu_uuid': uuid})
    return {'uuid': gpu[0], 'name': gpu[1], 'total_mib': float(gpu[2]), 'used_mib': float(gpu[3]),
            'utilization_percent': float(gpu[4]), 'processes': processes,
            'foreign': [p for p in processes if p['pid'] != os.getpid()]}


def host_memory():
    info = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        parts = line.split()
        if parts[0] in ('MemAvailable:', 'MemTotal:'): info[parts[0][:-1]] = int(parts[1])*1024
    return info


def gate(c, require_data=True):
    """Fails closed; returns concrete reasons. Does not invoke a solver or allocate volume arrays."""
    reasons = []; out = {'status': 'NOT EXECUTED', 'platform': platform.platform(), 'python': sys.executable,'config_sha256':obj_sha(c)}
    if platform.system() != 'Linux': reasons.append('baseline environment requires Linux')
    if Path(sys.executable).resolve() != Path(c['python']).resolve(): reasons.append('wrong Python executable')
    try: out['sources'] = source_hashes(c)
    except Exception as e: reasons.append(str(e))
    if require_data:
        try: out['inventory'] = input_inventory(c)
        except Exception as e: reasons.append(str(e))
    try:
        import numpy as np
        import cupy as cp
        import scipy
        import nibabel
        out['versions'] = {'numpy': np.__version__, 'cupy': cp.__version__, 'scipy': scipy.__version__, 'nibabel': nibabel.__version__,
                           'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(), 'cuda_driver': cp.cuda.runtime.driverGetVersion()}
        for k, v in {'numpy': '2.2.6', 'cupy': '14.2.0', 'scipy': '1.15.3', 'cuda_runtime': 13020,'cuda_driver':13020}.items():
            if out['versions'][k] != v: reasons.append(f'{k} differs from registered environment: expected {v}')
    except Exception as e: reasons.append(f'CUDA/package capability unavailable: {type(e).__name__}: {e}')
    try:
        gpu = smi(c); out['gpu'] = gpu
        if gpu['name'] != 'NVIDIA GeForce RTX 4090': reasons.append('requires NVIDIA GeForce RTX 4090')
        if gpu['foreign']: reasons.append('foreign GPU compute process present')
        out['host_memory'] = host_memory()
        out['cpu_model']=next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name'))
        baseline=readj(Path(c['e6_rows']).parent/'run_meta.json')
        out['python_version']=platform.python_version()
        if out['python_version']!=baseline['host']['python']: reasons.append('Python version differs from E6 baseline')
        if out['cpu_model']!=baseline['host']['cpu_model']: reasons.append('CPU model differs from E6 baseline')
        normal_uuid=lambda u:str(u).lower().removeprefix('gpu-').replace('-','')
        if normal_uuid(gpu['uuid'])!=normal_uuid(baseline['device'].get('gpu_uuid_cupy')): reasons.append('GPU UUID differs from E6 baseline')
    except Exception as e: reasons.append(f'resource inventory unavailable: {e}')
    out.update(ok=not reasons, reasons=reasons)
    return out


def admit(c, shape, K, rounds=2):
    from r8c_memory_model import model
    prediction = model(shape, K, c['delta'], rounds, reserve_factor=c['reserve_factor'], reserve_fixed_mib=c['reserve_fixed_mib'])
    gpu = smi(c); ram = host_memory()
    limit = min(c['vram_fraction']*gpu['total_mib']-gpu['used_mib'], gpu['total_mib']-gpu['used_mib']-c['headroom_mib'])
    # Host allowance includes current inputs, retained volume, validation and output I/O; still a safety scenario.
    host_need = prediction['host_scenario_bytes'] + 32*prediction['expanded_cells'] + 2**30
    why = []
    if gpu['foreign']: why.append('foreign GPU process present')
    if not prediction['fp_int32_ok']: why.append('FP int32 overflow')
    if not prediction['global_verifier']['ok']: why.append('global verifier REFUSED_INT32; primary design requires a global certificate')
    if prediction['pool_reserved_prediction_bytes']/2**20 > limit: why.append('predicted reserved GPU memory exceeds admission budget')
    if host_need > c['host_fraction']*ram['MemAvailable']: why.append('predicted host memory exceeds admission budget')
    return {'ok': not why, 'reasons': why, 'prediction': prediction, 'gpu': gpu, 'host': ram, 'vram_limit_mib': limit, 'host_need_bytes': host_need}


def seal(directory):
    d = Path(directory)
    lines = [f'{sha(p)}  {p.relative_to(d).as_posix()}' for p in sorted(d.rglob('*')) if p.is_file() and p != d/'SHA256SUMS.txt' and '__pycache__' not in p.parts]
    (d/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
