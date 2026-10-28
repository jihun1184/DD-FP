#!/usr/bin/env python3
"""Report the Python environment the GPU scripts will use (no GPU context is created). Usage: python r3_envprobe.py > env.json"""
import importlib, json, os, sys


def mod(name):
    try:
        m = importlib.import_module(name)
        return dict(version=str(getattr(m, "__version__", "?")), file=str(getattr(m, "__file__", "")))
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")


def main():
    rec = dict(python=sys.version.split()[0], executable=sys.executable, prefix=sys.prefix, base_prefix=getattr(sys, "base_prefix", sys.prefix),
               conda_env=os.environ.get("CONDA_DEFAULT_ENV"), virtual_env=os.environ.get("VIRTUAL_ENV"))
    rec["isolated_env"] = bool(sys.prefix != getattr(sys, "base_prefix", sys.prefix) or os.environ.get("CONDA_PREFIX"))
    rec["modules"] = {n: mod(n) for n in ("numpy", "scipy", "nibabel", "cupy", "tqdm")}
    if "version" in rec["modules"]["cupy"]:
        try:
            import cupy as cp
            rt = cp.cuda.runtime
            rec["cupy_cuda"] = dict(runtime_version=rt.runtimeGetVersion(), driver_version=rt.driverGetVersion(), device_count=rt.getDeviceCount())
            if rec["cupy_cuda"]["device_count"] > 0:
                pr = rt.getDeviceProperties(0)
                rec["cupy_cuda"]["device0"] = dict(name=pr["name"].decode(), cc=f"{pr['major']}.{pr['minor']}", total_mem_gib=round(pr["totalGlobalMem"] / 2**30, 2))
        except Exception as e:
            rec["cupy_cuda"] = dict(error=f"{type(e).__name__}: {e}")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
