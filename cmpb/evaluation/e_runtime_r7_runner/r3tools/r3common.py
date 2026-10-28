"""Shared helpers for the DD-FP R3 GPU server runner (stdlib only; runs under any Python >= 3.8).

Design rules (from the work order): no sudo, never touch other users' GPU processes, never overwrite inputs/earlier results,
keep every log and intermediate file even when a step fails, record hashes / GPU / CUDA / seeds / config for every run.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
PACKAGE_NAME = "DDFP_R3_GPU_SERVER_RUNNER_R0"

# ---- configuration --------------------------------------------------------------------------------------------------------------------
DEFAULTS = {
    "R3_PYTHON": "",                 # python of the isolated environment (venv/conda). empty -> <R3_VENV>/bin/python if it exists, else python3
    "R3_VENV": "",                   # default: <package parent>/ddfp_r3_venv
    "R3_REPO": "",                   # prepared DD-FP checkout (created by `prepare`). default: <package parent>/DD-FP-R3
    "R3_BRATS_ROOT": "",             # directory containing BraTS2021_*/*_flair.nii.gz (1,251 subjects)
    "R3_OUT_ROOT": "",               # results are written here, never inside the package. default: <package parent>/ddfp_r3_runs
    "R3_CUDA_DEVICE": "auto",        # physical GPU index (nvidia-smi order) or "auto" = first idle GPU
    "R3_REFERENCE_DIR": "",          # default: <package>/reference (recorded 4090 benchmark metadata + rows)
    "R3_PROTECT_DIRS": "",           # comma-separated directories that must never be written (e.g. old benchmark result folders)
    "R3_MIN_FREE_GPU_GIB": "8",
    "R3_MIN_DISK_GIB": "5",
    "R3_MIN_RAM_GIB": "12",
    "R3_IDLE_MEM_MIB": "1500",       # GPU memory already in use above this counts as "busy" (desktop/graphics usually < 1 GiB)
    "R3_IDLE_UTIL_PCT": "10",
    "R3_COHORT_SEED": "42",
    "R3_A_SEED": "20261014",
    "R3_A_N_RANDOM": "60",
    "R3_PILOT_SEQ": "0",             # 1 = also run the ~110 s sequential FP exact comparison in the pilot
    "R3_PILOT_TIMEOUT_S": "7200",
    "R3_STAGE1_TIMEOUT_S": "5400",
    "R3_TIMING_TOLERANCE": "0.30",   # pilot acceptance band around the recorded per-subject mean (same GPU model only)
    "R3_ALLOW_NON_ISOLATED_PYTHON": "0",
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def jwrite(path, obj) -> None:
    """Atomic JSON write (tmp + replace); refuses to write through a symlink."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def jread(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def parse_env_file(path) -> dict:
    """KEY=VALUE lines; '#' comments; optional quotes. Never executed by a shell."""
    out = {}
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        if "=" not in t:
            raise ValueError(f"{path}:{ln}: expected KEY=VALUE")
        k, v = t.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        elif " #" in v:
            v = v.split(" #", 1)[0].strip()
        if k not in DEFAULTS:
            raise ValueError(f"{path}:{ln}: unknown key {k!r} (allowed: {', '.join(sorted(DEFAULTS))})")
        out[k] = os.path.expandvars(os.path.expanduser(v))
    return out


def load_config(path=None, overrides=None) -> dict:
    cfg = dict(DEFAULTS)
    here = PKG.parent
    if path and Path(path).is_file():
        cfg.update(parse_env_file(path))
    for k, v in (overrides or {}).items():
        if v not in (None, ""):
            cfg[k] = str(v)
    cfg["R3_VENV"] = cfg["R3_VENV"] or str(here / "ddfp_r3_venv")
    cfg["R3_REPO"] = cfg["R3_REPO"] or str(here / "DD-FP-R3")
    cfg["R3_OUT_ROOT"] = cfg["R3_OUT_ROOT"] or str(here / "ddfp_r3_runs")
    cfg["R3_REFERENCE_DIR"] = cfg["R3_REFERENCE_DIR"] or str(PKG / "reference")
    if not cfg["R3_PYTHON"]:
        vp = Path(cfg["R3_VENV"]) / "bin" / "python"
        cfg["R3_PYTHON"] = str(vp) if vp.is_file() else (shutil.which("python3") or sys.executable)
    return cfg


def fnum(cfg, key, cast=float):
    try:
        return cast(cfg[key])
    except Exception:
        return cast(DEFAULTS[key])


# ---- package integrity ----------------------------------------------------------------------------------------------------------------
def manifest_entries(root=PKG):
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.name != "MANIFEST.sha256" and not p.name.endswith(".pyc"):
            yield p.relative_to(root).as_posix(), p


def verify_manifest(root=PKG) -> dict:
    mf = Path(root) / "MANIFEST.sha256"
    if not mf.is_file():
        return dict(ok=False, error="MANIFEST.sha256 missing", missing=[], changed=[], extra=[], n=0, manifest_sha256=None)
    want = {}
    for line in mf.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            want[rel] = h
    missing, changed = [], []
    for rel, h in want.items():
        p = Path(root) / rel
        if not p.is_file():
            missing.append(rel)
        elif sha256_file(p) != h:
            changed.append(rel)
    extra = [rel for rel, _ in manifest_entries(root) if rel not in want and not rel.startswith("selftest/_work/")]
    return dict(ok=not missing and not changed, missing=missing, changed=changed, extra=extra, n=len(want), manifest_sha256=sha256_file(mf))


# ---- subprocess with log preservation, timeout and optional monitor --------------------------------------------------------------------
def run_step(name, argv, logdir, env=None, cwd=None, timeout=None, monitor=None, interval=10.0, echo=True) -> dict:
    """Run argv, tee stdout+stderr to <logdir>/<name>.log, record to steps.tsv. Never raises; returns a record with rc/reason.

    monitor(pid) -> None | str : called every `interval` s while the child runs; a non-empty string terminates the child (SIGTERM, then SIGKILL).
    Only the child that this function started is ever signalled -- never any other process.
    """
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"{name}.log"
    rec = dict(name=name, argv=[str(a) for a in argv], log=str(log), start=now_iso(), rc=None, reason="", pid=None)
    t0 = time.time()
    interrupted, p = False, None
    try:
        with open(log, "ab", buffering=0) as lf:
            lf.write(f"# {rec['start']} :: {' '.join(rec['argv'])}\n".encode())
            p = subprocess.Popen(rec["argv"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd, bufsize=0)
            rec["pid"] = p.pid

            def pump():
                for chunk in iter(lambda: p.stdout.read(4096), b""):
                    lf.write(chunk)
                    if echo:
                        try:
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.buffer.flush()
                        except Exception:
                            pass

            th = threading.Thread(target=pump, daemon=True)
            th.start()
            last = t0
            while p.poll() is None:
                time.sleep(0.5)
                el = time.time() - t0
                if timeout and el > timeout and not rec["reason"]:
                    rec["reason"] = f"TIMEOUT after {int(el)} s (limit {int(timeout)} s)"
                elif monitor and time.time() - last >= interval and not rec["reason"]:
                    last = time.time()
                    try:
                        why = monitor(p.pid)
                    except Exception as e:  # a failing monitor must not kill the run silently
                        why = None
                        lf.write(f"# monitor error {type(e).__name__}: {e}\n".encode())
                    if why:
                        rec["reason"] = str(why)
                if rec["reason"]:
                    lf.write(f"# terminating own child pid {p.pid}: {rec['reason']}\n".encode())
                    try:
                        p.send_signal(signal.SIGTERM)
                        try:
                            p.wait(20)
                        except subprocess.TimeoutExpired:
                            p.kill()
                    except Exception:
                        pass
                    break
            p.wait()
            th.join(5)
            rec["rc"] = p.returncode
    except KeyboardInterrupt:                                  # Ctrl-C: stop OUR child only, keep the log, record the step, then re-raise
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(20)
                except subprocess.TimeoutExpired:
                    p.kill()
        except Exception:
            pass
        rec["rc"], rec["reason"], interrupted = 130, "interrupted by user (Ctrl-C)", True
    except FileNotFoundError as e:
        rec["rc"], rec["reason"] = 127, f"command not found: {e}"
    except Exception as e:
        rec["rc"], rec["reason"] = 126, f"{type(e).__name__}: {e}"
    rec["end"], rec["seconds"] = now_iso(), round(time.time() - t0, 2)
    with open(logdir / "steps.tsv", "a", encoding="utf-8") as f:
        f.write("\t".join([rec["name"], str(rec["rc"]), rec["start"], rec["end"], str(rec["seconds"]), rec["reason"] or "-", " ".join(rec["argv"])]) + "\n")
    if interrupted:
        raise KeyboardInterrupt
    return rec


# ---- GPU state (nvidia-smi only; read-only) ----------------------------------------------------------------------------------------------
def smi(args, timeout=20):
    try:
        r = subprocess.run(["nvidia-smi"] + list(args), capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except FileNotFoundError:
        return 127, "nvidia-smi not found"
    except Exception as e:
        return 126, f"{type(e).__name__}: {e}"


def _csv(text):
    return [[c.strip() for c in ln.split(",")] for ln in text.splitlines() if ln.strip()]


def gpu_list() -> list:
    rc, out = smi(["--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,power.draw,power.limit,pci.bus_id", "--format=csv,noheader,nounits"])
    if rc != 0:
        return []
    keys = ["index", "uuid", "name", "driver", "mem_total_mib", "mem_used_mib", "util_pct", "power_w", "power_limit_w", "pci"]
    return [dict(zip(keys, row)) for row in _csv(out) if len(row) == len(keys)]


def compute_apps() -> list:
    rc, out = smi(["--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    if rc != 0 or "No running" in out:
        return []
    return [dict(zip(["uuid", "pid", "name", "mem_mib"], r)) for r in _csv(out) if len(r) == 4]


def gpu_status(index, own_pids=(), min_free_gib=8.0, idle_mem_mib=1500.0, idle_util=10.0, samples=3) -> dict:
    """Decide whether physical GPU `index` may be used. Pure observation.

    Foreign process = compute app on this GPU whose pid is not ours. Inside containers pids may be translated; if none of the visible apps
    carries one of our pids, a single visible app is assumed to be ours and two or more count as foreign.
    """
    gpus = gpu_list()
    g = next((x for x in gpus if x["index"] == str(index)), None)
    if g is None:
        return dict(status="NO_GPU", reasons=[f"GPU index {index} not visible to nvidia-smi"], gpu=None, foreign=[])
    own = {str(p) for p in own_pids}
    apps = [a for a in compute_apps() if a["uuid"] == g["uuid"]]
    visible_own = [a for a in apps if a["pid"] in own]
    foreign = [a for a in apps if a["pid"] not in own]
    if own and not visible_own and foreign:
        foreign = foreign[1:]
    reasons = []
    if foreign:
        reasons.append("other compute process(es): " + "; ".join(f"pid {a['pid']} {a['name']} {a['mem_mib']} MiB" for a in foreign))
    utils = []
    for i in range(samples):
        _, o = smi(["-i", str(index), "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
        row = _csv(o)
        if row and len(row[0]) == 3:
            utils.append((float(row[0][0]), float(row[0][1]), float(row[0][2])))
        if i < samples - 1:
            time.sleep(0.5)
    mem_used = max([u[1] for u in utils] or [float(g["mem_used_mib"] or 0)])
    mem_tot = (utils[-1][2] if utils else float(g["mem_total_mib"] or 0))
    util = max([u[0] for u in utils] or [0.0])
    if not own:                                    # start-of-run test: the GPU must already be quiet
        if mem_used > idle_mem_mib:
            reasons.append(f"{mem_used:.0f} MiB already allocated (> {idle_mem_mib:.0f} MiB idle limit)")
        if util > idle_util:
            reasons.append(f"GPU utilisation {util:.0f} % (> {idle_util:.0f} %)")
    free_gib = (mem_tot - mem_used) / 1024.0
    if free_gib < min_free_gib:
        reasons.append(f"only {free_gib:.1f} GiB free (< {min_free_gib:.1f})")
    return dict(status="BUSY" if reasons else "IDLE", reasons=reasons, gpu=g, foreign=foreign, mem_used_mib=mem_used, util_pct=util, free_gib=round(free_gib, 2))


def pick_gpu(spec, **kw) -> dict:
    """spec = index string or 'auto' (first idle GPU, preferring names containing '4090')."""
    gpus = gpu_list()
    if not gpus:
        return dict(status="NO_GPU", reasons=["nvidia-smi reports no GPU"], gpu=None, foreign=[])
    if spec != "auto":
        return gpu_status(spec, **kw)
    cands = sorted(gpus, key=lambda g: ("4090" not in g["name"], int(g["index"])))
    last = None
    for g in cands:
        st = gpu_status(g["index"], **kw)
        if st["status"] == "IDLE":
            return st
        last = st
    last = last or dict(status="NO_GPU", reasons=[], gpu=None, foreign=[])
    last["reasons"] = ["no idle GPU found (auto): " + "; ".join(last.get("reasons", []))]
    return last


# ---- misc host facts --------------------------------------------------------------------------------------------------------------------
def host_facts() -> dict:
    f = dict(hostname=platform.node(), platform=platform.platform(), python=sys.version.split()[0], python_exe=sys.executable)
    try:
        f["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if ":" in l and l.split()[1].isdigit()}
        f["mem_total_gib"] = round(mi.get("MemTotal", 0) / 2**20, 1)
        f["mem_available_gib"] = round(mi.get("MemAvailable", 0) / 2**20, 1)
        f["loadavg"] = os.getloadavg()
        f["cpu_count"] = os.cpu_count()
    except Exception as e:
        f["probe_error"] = f"{type(e).__name__}: {e}"
    return f


def disk_free_gib(path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return round(shutil.disk_usage(p).free / 2**30, 1)


def is_inside(child, parent) -> bool:
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except Exception:
        return False


def check_output_location(cfg) -> list:
    """Reasons why R3_OUT_ROOT is unsafe (would risk overwriting data). Empty list = fine."""
    out = Path(cfg["R3_OUT_ROOT"]).expanduser()
    bad = []
    prot = [p for p in cfg.get("R3_PROTECT_DIRS", "").split(",") if p.strip()] + [cfg.get("R3_BRATS_ROOT", "")]
    for p in prot:
        if p and Path(p).exists() and (is_inside(out, p) or is_inside(p, out)):
            bad.append(f"R3_OUT_ROOT {out} overlaps protected/data directory {p}")
    if is_inside(out, PKG) or is_inside(PKG, out):
        bad.append("R3_OUT_ROOT must not overlap the runner package (it would break MANIFEST verification)")
    if out.exists() and any((out / n).exists() for n in ("benchmark_rows.csv", "benchmark_metadata.json")):
        bad.append(f"{out} contains benchmark files -- refusing to write results next to the recorded benchmark")
    return bad


def child_env(cfg, gpu_index=None) -> dict:
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env
