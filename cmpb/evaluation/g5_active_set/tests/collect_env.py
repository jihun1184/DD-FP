#!/usr/bin/env python3
"""collect_env.py -- record the execution environment of a GPU run (G3 recorded only GPU/cupy/CUDA-runtime/driver version numbers).
Run on the server in the SAME python environment as the runner, once per run directory:
  python3 -I tests/collect_env.py --out RUN_DIR/environment.json [--files RUN_DIR/summary.json ...]
Writes: python/OS/CPU/RAM, package versions (numpy, cupy, nibabel, scipy), CUDA runtime/driver/NVRTC versions as seen by cupy, nvidia-smi (driver, GPU, clocks, persistence,
other compute processes), the git/zip hash of nothing it cannot see, and SHA-256 of the files named in --files and of the code in this work tree."""
import argparse, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd, t=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=t).stdout.strip()
    except Exception as e:
        return f"unavailable: {type(e).__name__}"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--files", nargs="*", default=[])
    A = ap.parse_args()
    E = dict(collected=time.strftime("%Y-%m-%dT%H:%M:%S%z"), python=sys.version, executable=sys.executable, platform=platform.platform(), machine=platform.machine(),
             cpu=sh(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo || true"]), nproc=os.cpu_count(), mem=sh(["bash", "-c", "grep -E 'MemTotal|SwapTotal' /proc/meminfo || true"]),
             env={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUDA_PATH", "CUPY_ACCELERATORS", "CUPY_CACHE_DIR", "OMP_NUM_THREADS", "PYTHONHASHSEED")})
    for mod in ("numpy", "cupy", "nibabel", "scipy"):
        try:
            m = __import__(mod); E.setdefault("packages", {})[mod] = getattr(m, "__version__", "?")
        except Exception as e:
            E.setdefault("packages", {})[mod] = f"not importable: {type(e).__name__}"
    try:
        import cupy as cp
        E["cuda"] = dict(runtime=cp.cuda.runtime.runtimeGetVersion(), driver=cp.cuda.runtime.driverGetVersion(), device0=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                         nvrtc=getattr(cp.cuda.nvrtc, "getVersion", lambda: "?")())
    except Exception as e:
        E["cuda"] = f"cupy unavailable: {type(e).__name__}: {e}"
    E["nvidia_smi_L"] = sh(["nvidia-smi", "-L"])
    E["nvidia_smi_query"] = sh(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,pstate,clocks.sm,clocks.max.sm,clocks.mem,persistence_mode,power.limit,memory.total,memory.used,temperature.gpu", "--format=csv"])
    E["nvidia_smi_compute_apps"] = sh(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"])
    E["code_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in sorted(list((ROOT / "src").glob("*.py")) + list((ROOT / "tests").glob("*.py")) + list((ROOT / "frozen").glob("*.py")))}
    E["file_sha256"] = {f: sha(f) for f in A.files if Path(f).is_file()}
    A.out.write_text(json.dumps(E, indent=1))
    print(json.dumps({k: E[k] for k in ("python", "platform", "packages", "cuda", "nvidia_smi_L")}, indent=1))


if __name__ == "__main__":
    main()
