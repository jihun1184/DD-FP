"""R7 shared helpers (numpy; cupy only for the real backend).  Verification code: it IMPORTS the frozen production modules, never edits them.

Backends
  cupy : real CUDA path (production gpu_immersion / gpu_seed_aware_worklist / experiment_DDFP_all).  cuda_executed = True only here.
  stub : CPU stand-ins (seedref state machine, lib/r1_check53 checker, production scheduler in shim mode).  PLUMBING TEST ONLY.
         Every record written by a stub run carries backend="stub", cuda_executed=false; the verifier refuses it as GPU evidence.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
R7_VERSION = "R7-A.1"
EXPECTED = {"scripts/ddfp/experiment_DDFP_all.py": "b6a9b109f4105e7b77ebed880f669ba670bbd87123deca9509ba5b00993e561e",
            "src/ddfp/cpu_fp.py": "8eae554f7d3d688dee2285f41d20bfbf6e34af98016a3f1ad07f1ff5f8f36a88",
            "src/ddfp/gpu_immersion.py": "b784a376d3466a1d4ded5089a2eee9f27df35006b4926301dd06106b4ed262a4",
            "src/ddfp/gpu_native_worklist.py": "853bf33a6bbd9c91f5304a018f0a51ff53e8418360fe1477208cf81df17163ac",
            "src/ddfp/gpu_seed_aware_worklist.py": "c80f56b9ba7afd63196332d51894168d36f6e9f3e2a51b6b966786311e4eecad",
            "test/test_smoke.py": "d1dea404932f1b022830a5c21cb15daae67e480d5f2d54a882d2b98ac7050811"}
METHODS = ("gpu_k1", "ddfp_k16_d1", "seq_fp_cpu")


class Die(SystemExit):
    """SystemExit with a message: 2 = environment/integrity, 3 = internal, 4 = GPU busy."""
    def __init__(self, code, msg):
        print("ERROR:", msg, file=sys.stderr, flush=True)
        super().__init__(code)


def sha_b(b: bytes) -> str: return hashlib.sha256(b).hexdigest()


def sha_f(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def arr_sha(a) -> str: return sha_b(np.ascontiguousarray(a).tobytes())


def sample_hash(a) -> str:      # identical to the recorded benchmark's 1/4096-strided sample hash
    return sha_b(np.ascontiguousarray(a.ravel()[::4096]).view(np.uint8))


def f32(a): return np.ascontiguousarray(a, np.float32)


def cmp_arrays(a, b, keep=20) -> dict:
    """Bitwise comparison of two float32 arrays (+0/-0 and NaN payloads count as different bits).
    n_diff_bits: positions with different bit patterns; n_diff_value: positions with a != b (NaN==NaN treated equal);
    max_abs_diff over positions where both are finite and the values differ (float64)."""
    a, b = f32(a), f32(b)
    if a.shape != b.shape:
        return dict(shape_equal=False, shape_a=list(a.shape), shape_b=list(b.shape), n_diff_bits=-1, n_diff_value=-1, max_abs_diff=None, equal=False)
    ua, ub = a.view(np.uint32), b.view(np.uint32)
    d = ua != ub
    n = int(d.sum())
    out = dict(shape_equal=True, n_cells=int(a.size), n_diff_bits=n, n_nan_a=int(np.isnan(a).sum()), n_nan_b=int(np.isnan(b).sum()), equal=(n == 0))
    if n:
        both = d & np.isfinite(a) & np.isfinite(b)
        va = (a != b) & ~(np.isnan(a) & np.isnan(b))
        out["n_diff_value"] = int(va.sum())
        out["max_abs_diff"] = float(np.max(np.abs(a[both].astype(np.float64) - b[both].astype(np.float64)))) if both.any() else None
        idx = np.argwhere(d)[:keep]
        out["first_diffs"] = [dict(idx=[int(x) for x in i], a=float(a[tuple(i)]), b=float(b[tuple(i)])) for i in idx]
    else:
        out["n_diff_value"], out["max_abs_diff"] = 0, 0.0
    return out


# ---- host memory (zero-perturbation): VmHWM reset via /proc/self/clear_refs ---------------------------------------------------------------
def hwm_reset() -> bool:
    try:
        with open("/proc/self/clear_refs", "w") as f: f.write("5")
        return True
    except Exception:
        return False


def rss_mib() -> dict:
    out = {}
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith(("VmHWM", "VmRSS")): out[ln.split(":")[0]] = round(int(ln.split()[1]) / 1024, 1)
    except Exception:
        pass
    return out


def host_facts() -> dict:
    f = dict(platform=__import__("platform").platform(), python=sys.version.split()[0], numpy=np.__version__, cpu_count=os.cpu_count())
    try:
        f["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if ":" in l and l.split()[1].isdigit()}
        f["mem_total_gib"], f["mem_available_gib"] = round(mi["MemTotal"] / 2**20, 1), round(mi["MemAvailable"] / 2**20, 1)
        f["loadavg"] = os.getloadavg()
    except Exception as e:
        f["probe_error"] = f"{type(e).__name__}: {e}"
    return f


# ---- nvidia-smi (read-only) -------------------------------------------------------------------------------------------------------------------
def nvsmi(query: str, gpu_index=None, fmt="csv,noheader,nounits") -> str:
    cmd = ["nvidia-smi"] + (["-i", str(gpu_index)] if gpu_index is not None else []) + [f"--query-{query}", f"--format={fmt}"]
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e: return f"UNAVAILABLE({type(e).__name__})"


def foreign_procs(gpu_index=None) -> list:
    txt = nvsmi("compute-apps=pid,process_name,used_memory", gpu_index)
    if txt.startswith("UNAVAILABLE"): return ["UNAVAILABLE"]
    return [l for l in txt.splitlines() if l.strip() and l.split(",")[0].strip() != str(os.getpid())]


def gpu_snapshot(gpu_index=None) -> dict:
    return dict(smi=nvsmi("gpu=index,name,uuid,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu", gpu_index), foreign=foreign_procs(gpu_index))


class Sampler:
    """Polling sampler for device memory (cudaMemGetInfo) and host RSS.  Used only in NON-timed runs (E4, E6 memory pass).
    peak_vram_delta_mib = max(used) - used_at_start.  Scope: from __enter__ to __exit__ (caller wraps exactly the algorithm call)."""

    def __init__(self, backend, interval=0.02):
        self.b, self.dt, self.stop, self.peak_dev, self.peak_rss, self.n, self.base = backend, interval, threading.Event(), 0.0, 0.0, 0, 0.0
        self.th = None

    def _loop(self):
        try:
            self.b.thread_init()
            while not self.stop.is_set():
                self.peak_dev = max(self.peak_dev, self.b.device_used_mib())
                self.peak_rss = max(self.peak_rss, rss_mib().get("VmRSS", 0.0)); self.n += 1
                time.sleep(self.dt)
        except Exception as e:
            self.err = f"{type(e).__name__}: {e}"

    def __enter__(self):
        self.err = None; self.base = self.b.device_used_mib(); self.peak_dev = self.base
        self.th = threading.Thread(target=self._loop, daemon=True); self.th.start(); return self

    def __exit__(self, *a):
        self.peak_dev = max(self.peak_dev, self.b.device_used_mib()); self.stop.set(); self.th.join(5)

    def result(self) -> dict:
        return dict(peak_device_used_mib=round(self.peak_dev, 1), baseline_device_used_mib=round(self.base, 1), peak_vram_delta_mib=round(self.peak_dev - self.base, 1),
                    peak_rss_mib=round(self.peak_rss, 1), n_samples=self.n, interval_s=self.dt, sampler_error=self.err,
                    scope="from entering the algorithm call to leaving it (inputs already on host; verification excluded)")


# ---- importing frozen R3 helpers by path -----------------------------------------------------------------------------------------------------
def load_by_path(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def r3_global():      # sha_f / sample_hash / select_cohort exactly as in the frozen R3-B script
    return load_by_path("r3_global_check_timing", PKG / "r3_scripts" / "r3_global_check_timing.py")


def cohort(brats_root, metadata_path, n, seed):
    G = r3_global()
    meta = json.loads(Path(metadata_path).read_text()) if metadata_path else None
    paths = G.select_cohort(brats_root, n, seed, meta)
    return [(p.parent.name, p) for p in paths], meta


def synthetic_cohort(n, size, seed=7):
    rng = np.random.default_rng(seed)
    return [(f"synthetic{i}", None, rng.integers(0, 256, (size,) * 3, dtype=np.uint8)) for i in range(n)]


# ---- backends ----------------------------------------------------------------------------------------------------------------------------------
class _Base:
    cuda = False

    def thread_init(self): pass
    def sync(self): pass
    def free(self): pass
    def device_used_mib(self): return 0.0
    def memstats(self): return {}
    def device_info(self): return dict(backend=self.name, cuda_executed=self.cuda)


class StubBackend(_Base):
    """CPU stand-ins.  fault in {None,'parity','dwc'} injects a deterministic defect used only by the negative controls."""
    name = "stub"

    def __init__(self, repo, lib, fault=None, K=3):
        sys.path.insert(0, str(lib))
        import cpu_fp, seedref, r1_check53
        self.cpu_fp, self.seedref, self.r1, self.fault, self.K = cpu_fp, seedref, r1_check53, fault, K
        D = load_by_path("r3_diff_ibi", PKG / "r3_scripts" / "r3_diff_ibi.py")
        self.prod = D.load_production(Path(repo), "shim", cpu_fp, seedref, "lower")
        self.preprocess = self.prod.preprocess

    def prep(self, vol): return self.cpu_fp.build_ispan_cpu(vol)
    def ispan_host(self, lo, hi): return lo, hi
    def fp_k1(self, lo, hi, l): return f32(self.seedref.seed_aware_cpu_reference(lo, hi, l, None, "lower")[0])

    def fp_tie(self, lo, hi, l, tie):
        out, meta, _ = self.seedref.seed_aware_cpu_reference(lo, hi, l, None, tie)
        out = f32(out).copy()
        if self.fault == "parity" and tie == "higher":
            out[out.shape[0] // 2, 1, 1] += 1.0
        return out, meta

    def ibi(self, vol, K, delta, max_rounds):
        r = self.prod.run_ibi_v10(vol, K=self.K, delta=delta, max_rounds=max_rounds)
        return dict(u_dd=f32(r["u_dd"]), R_star=int(r["R_star"]), boundary_z_orig=list(r["boundary_z_orig"]))

    def seam(self, u, bz): return int(self.prod.count_boundary_violations(u, bz))
    def h2d(self, u): return u

    def global_check(self, u):
        v = int(sum(self.r1.check(f32(u), slab=16)[0].values()))
        if self.fault == "dwc": v += 1
        return dict(dwc_ok=v == 0, violations=v, nan_skipped=0, total_blocks=-1)

    def cpu_check(self, u): return self.r1.check(f32(u), slab=16)[0]


class CupyBackend(_Base):
    name, cuda = "cupy", True

    def __init__(self, repo, lib, fault=None, K=16):
        repo = Path(repo)
        bad = {k: sha_f(repo / k) for k, v in EXPECTED.items() if not (repo / k).is_file() or sha_f(repo / k) != v}
        if bad: raise Die(2, f"production hash mismatch: {sorted(bad)}")
        sys.path.insert(0, str(repo)); sys.path.insert(0, str(lib))
        import cupy as cp
        import r1_check53
        from scripts.ddfp.experiment_DDFP_all import preprocess, run_ibi_v10, count_boundary_violations
        from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu, verify_dwc_gpu
        from src.ddfp.gpu_seed_aware_worklist import seed_aware_boutry_bucket_gpu
        self.cp, self.r1, self.K = cp, r1_check53, K
        self.preprocess, self._run_ibi, self._seam, self._build = preprocess, run_ibi_v10, count_boundary_violations, build_ispan_gpu
        self._fp, self._verify, self._tie = front_propagation_gpu, verify_dwc_gpu, seed_aware_boutry_bucket_gpu
        cp.cuda.Device(0).use(); cp.zeros(1).sum()              # context creation (device 0 = CUDA_VISIBLE_DEVICES-selected GPU)

    def thread_init(self): self.cp.cuda.Device(0).use()
    def sync(self): self.cp.cuda.Stream.null.synchronize()
    def free(self): self.cp.get_default_memory_pool().free_all_blocks()

    def device_used_mib(self):
        f, t = self.cp.cuda.runtime.memGetInfo(); return (t - f) / 2**20

    def memstats(self):
        p = self.cp.get_default_memory_pool(); f, t = self.cp.cuda.runtime.memGetInfo()
        return dict(pool_total_mib=round(p.total_bytes() / 2**20, 1), pool_used_mib=round(p.used_bytes() / 2**20, 1), device_used_mib=round((t - f) / 2**20, 1), device_total_mib=round(t / 2**20, 1))

    def device_info(self):
        pr = self.cp.cuda.runtime.getDeviceProperties(0)
        uu = pr.get("uuid", b"")
        return dict(backend="cupy", cuda_executed=True, gpu_name=pr["name"].decode() if isinstance(pr["name"], bytes) else str(pr["name"]), gpu_uuid_cupy=(bytes(uu).hex() if uu else None),
                    cc=f"{pr['major']}.{pr['minor']}", total_mem_gib=round(pr["totalGlobalMem"] / 2**30, 2), cupy=self.cp.__version__, cuda_runtime=self.cp.cuda.runtime.runtimeGetVersion(),
                    cuda_driver=self.cp.cuda.runtime.driverGetVersion())

    def prep(self, vol): return self._build(vol)
    def ispan_host(self, lo, hi): return self.cp.asnumpy(lo), self.cp.asnumpy(hi)
    def fp_k1(self, lo, hi, l): return f32(self.cp.asnumpy(self._fp(lo, hi, l, verbose=False)))

    def fp_tie(self, lo, hi, l, tie):
        out, meta, _ = self._tie(self.cp, lo, hi, l, None, tie)
        return f32(self.cp.asnumpy(out) if not isinstance(out, np.ndarray) else out), meta

    def ibi(self, vol, K, delta, max_rounds):
        r = self._run_ibi(vol, K=K, delta=delta, max_rounds=max_rounds)
        return dict(u_dd=f32(r["u_dd"]), R_star=int(r["R_star"]), boundary_z_orig=list(r["boundary_z_orig"]))

    def seam(self, u, bz): return int(self._seam(u, bz))
    def h2d(self, u): return self.cp.asarray(u)
    def global_check(self, ug): return self._verify(ug, verbose=False)
    def cpu_check(self, u): return self.r1.check(f32(u), slab=16)[0]


def make_backend(kind, repo, lib, fault=None):
    if kind == "cupy": return CupyBackend(repo, lib, fault)
    if kind == "stub": return StubBackend(repo, lib, fault)
    raise Die(2, f"unknown backend {kind}")


def meta_light(meta: dict) -> dict:
    keys = ("tie_policy", "equidistant_next_level_tie_events", "scheduler_transition_count", "backward_scheduler_transition_count", "total_activation_rounds",
            "nan_count", "discovered_count", "processed_count", "pending_total_final", "schedule_direction_counts")
    out = {k: meta.get(k) for k in keys}
    lt = meta.get("level_transitions")
    if lt is not None: out["level_transitions_len"], out["level_transitions_sha256"] = len(lt), sha_b(json.dumps(lt).encode())
    return out


def gpu_guard(backend, gpu_index, min_free_gib, allow_shared):
    """Refuse to start when another process uses the GPU (never signals anybody)."""
    if backend != "cupy": return dict(guard="stub: n/a")
    f = foreign_procs(gpu_index)
    info = dict(foreign_before=f, smi=nvsmi("gpu=index,name,uuid,memory.used,memory.total,utilization.gpu", gpu_index))
    if f == ["UNAVAILABLE"] and not allow_shared: raise Die(2, "nvidia-smi unavailable: cannot verify that no other job uses the GPU")
    if f and f != ["UNAVAILABLE"] and not allow_shared: raise Die(4, f"GPU in use by other process(es): {f!r} -- refusing to start")
    return info


def jdump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp"); tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n"); os.replace(tmp, path)


def append_jsonl(path, obj):
    with open(path, "a") as f: f.write(json.dumps(obj, default=str) + "\n"); f.flush(); os.fsync(f.fileno())
