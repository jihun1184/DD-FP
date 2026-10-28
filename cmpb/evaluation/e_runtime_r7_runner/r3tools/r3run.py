#!/usr/bin/env python3
"""DD-FP R3 GPU server runner -- orchestrator (stdlib only, Python >= 3.8).  Use through ./run.sh.

  stage0   environment check (default).  Read-only except a few-second 12^3 CUDA wiring probe on a GPU the guard found idle.
  prepare  build the patched production checkout from the bundled git bundle + PATCH.diff (offline) and verify the 6 frozen hashes.
  stage1   P3 CUDA parity (existing gate) + script A (run_ibi_v10 vs CPU port) on the GPU.     needs stage0 PASS
  stage2   N=1 pilot of script B: global face+cube check and cost split.                       needs stage1 PASS
  stage3   N=20 cohort of script B.                                                            needs stage2 PASS and  --confirm N20
  status   list attempts / verdicts of the current run.   verify   check the package manifest.   setup-venv  create an isolated environment (explicit).
Nothing here uses sudo, signals foreign processes, or writes inside the package, the data directory or earlier result folders.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3common import (PKG, PACKAGE_NAME, check_output_location, child_env, disk_free_gib, fnum, gpu_list, gpu_status, host_facts, jread, jwrite,  # noqa: E402
                      load_config, now_iso, pick_gpu, run_step, sha256_file, smi, stamp, verify_manifest)

TOOLS, SCRIPTS = PKG / "r3tools", PKG / "r3_scripts"
BASE = "8fdfe6233dfe53f996ed2f09e8810bca53d5323b"
FROZEN = {"scripts/ddfp/experiment_DDFP_all.py": "b6a9b109f4105e7b77ebed880f669ba670bbd87123deca9509ba5b00993e561e",
          "src/ddfp/cpu_fp.py": "8eae554f7d3d688dee2285f41d20bfbf6e34af98016a3f1ad07f1ff5f8f36a88",
          "src/ddfp/gpu_immersion.py": "b784a376d3466a1d4ded5089a2eee9f27df35006b4926301dd06106b4ed262a4",
          "src/ddfp/gpu_native_worklist.py": "853bf33a6bbd9c91f5304a018f0a51ff53e8418360fe1477208cf81df17163ac",
          "src/ddfp/gpu_seed_aware_worklist.py": "c80f56b9ba7afd63196332d51894168d36f6e9f3e2a51b6b966786311e4eecad",
          "test/test_smoke.py": "d1dea404932f1b022830a5c21cb15daae67e480d5f2d54a882d2b98ac7050811"}
EVAL_NAMES = {"cpu_fp.py": "src/ddfp/cpu_fp.py", "experiment_DDFP_all.py": "scripts/ddfp/experiment_DDFP_all.py", "gpu_immersion.py": "src/ddfp/gpu_immersion.py",
              "gpu_native_worklist.py": "src/ddfp/gpu_native_worklist.py", "gpu_seed_aware_worklist.py": "src/ddfp/gpu_seed_aware_worklist.py", "test_smoke.py": "test/test_smoke.py"}
EXPECTED_PATHS = ["src/ddfp/cpu_fp.py", "src/ddfp/gpu_immersion.py", "src/ddfp/gpu_native_worklist.py", "src/ddfp/gpu_seed_aware_worklist.py", "test/test_smoke.py"]


# ------------------------------------------------------------------------------------------------------------------ context
class Ctx:
    pass


def make_ctx(a) -> Ctx:
    c = Ctx()
    cfgfile = a.config or (PKG / "config" / "r3.env")
    c.cfg = load_config(cfgfile, dict(R3_BRATS_ROOT=a.brats_root, R3_OUT_ROOT=a.out_root, R3_REPO=a.repo, R3_CUDA_DEVICE=a.gpu, R3_PYTHON=a.python))
    c.cfg_file = str(cfgfile) if Path(cfgfile).is_file() else "(none: defaults + command line)"
    c.smoke, c.args = bool(a.smoke), a
    c.out_root = Path(c.cfg["R3_OUT_ROOT"]).expanduser()
    c.py, c.repo = c.cfg["R3_PYTHON"], Path(c.cfg["R3_REPO"]).expanduser()
    c.ref = Path(c.cfg["R3_REFERENCE_DIR"])
    c.ptr = c.out_root / ("CURRENT_RUN_SMOKE" if c.smoke else "CURRENT_RUN")
    return c


def say(msg=""):
    print(msg, flush=True)


def run_dir_for(c, create: bool) -> Path:
    prefix = "SMOKE_" if c.smoke else ""
    if create:
        bad = check_output_location(c.cfg)
        if bad:
            raise SystemExit("ERROR: " + "; ".join(bad))
        c.out_root.mkdir(parents=True, exist_ok=True)
        rid = c.args.run_id or ((c.ptr.read_text().strip() if c.ptr.is_file() and not c.args.new_run else "") or f"{prefix}r3_{stamp()}")
        if not rid.startswith(prefix):
            rid = prefix + rid
        c.ptr.write_text(rid + "\n")
    else:
        rid = c.args.run_id or (c.ptr.read_text().strip() if c.ptr.is_file() else "")
        if not rid:
            raise SystemExit("ERROR: no current run. Run `./run.sh stage0` first." + (" (smoke)" if c.smoke else ""))
    d = c.out_root / rid
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_attempt(rd: Path, stage: str) -> Path:
    at = rd / stage / f"attempt_{stamp()}"
    n = 0
    while at.exists():                         # never overwrite an earlier attempt
        n += 1
        at = rd / stage / f"attempt_{stamp()}_{n}"
    at.mkdir(parents=True)
    return at


def attempts(rd: Path, stage: str):
    return sorted((rd / stage).glob("attempt_*"), reverse=True) if (rd / stage).is_dir() else []


# ------------------------------------------------------------------------------------------------------------------ gates
def gate_file(c):
    return "SMOKE_PASS.json" if c.smoke else "PASS.json"


def find_pass(c, rd, stage):
    for at in attempts(rd, stage):
        p = at / gate_file(c)
        if p.is_file():
            return at, jread(p)
    return None, None


def gate_check(c, rd, stage):
    """The previous stage must have a PASS record for the SAME package manifest, GPU and (still intact) verdict file."""
    at, rec = find_pass(c, rd, stage)
    if rec is None:
        return False, f"{stage} has no {gate_file(c)} in {rd}. Run `./run.sh {stage}` until it passes.", None
    man = verify_manifest()
    if not man["ok"] or rec.get("manifest_sha256") != man["manifest_sha256"]:
        return False, f"package changed since {stage} passed (manifest mismatch) -- results from different code must not be mixed. Re-run from stage0.", rec
    v = at / "verdict.json"
    if not v.is_file() or sha256_file(v) != rec.get("verdict_sha256"):
        return False, f"{stage} verdict file was modified or removed.", rec
    if not c.smoke:
        g = next((x for x in gpu_list() if x["index"] == str(rec.get("gpu_index"))), None)
        if g is None or g["uuid"] != rec.get("gpu_uuid"):
            return False, f"GPU {rec.get('gpu_index')} ({str(rec.get('gpu_uuid'))[:12]}...) of {stage} is not visible any more / changed. Re-run stage0.", rec
    rec["_attempt"] = str(at)
    return True, f"{stage} PASS ({at.name})", rec


def write_pass(c, at, stage, gpu=None):
    v = jread(at / "verdict.json", {})
    ok = (v.get("verdict") == "SMOKE_PASS") if c.smoke else (v.get("verdict") == "PASS" and v.get("evidence_class") == "GPU")
    if stage == "stage0" and not c.smoke:
        ok = v.get("verdict") == "PASS"
    if not ok:
        return False
    jwrite(at / gate_file(c), dict(stage=stage, created=now_iso(), manifest_sha256=verify_manifest()["manifest_sha256"], verdict_sha256=sha256_file(at / "verdict.json"),
                                    evidence_class=v.get("evidence_class"), gpu_index=(gpu or {}).get("index"), gpu_uuid=(gpu or {}).get("uuid"), gpu_name=(gpu or {}).get("name")))
    return True


def seal(at: Path):
    """Hash every file of the attempt so later edits are detectable."""
    lines = []
    for p in sorted(at.rglob("*")):
        if p.is_file() and p.name != "ARTIFACT_SHA256.txt":
            lines.append(f"{sha256_file(p)}  {p.relative_to(at).as_posix()}")
    (at / "ARTIFACT_SHA256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------------ repo / meta
def git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).rstrip("\n")


def repo_state(repo: Path) -> dict:
    st = dict(repo=str(repo), exists=repo.is_dir(), head=None, changed_paths=[], hashes={}, ok=False, problems=[])
    if not (repo / ".git").exists():
        st["problems"].append("not a git checkout (run `./run.sh prepare`)")
        return st
    rc, head = git(repo, "rev-parse", "HEAD")
    st["head"] = head if rc == 0 else None
    rc, out = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    st["changed_paths"] = sorted(l[3:] for l in out.splitlines() if len(l) > 3) if rc == 0 else []
    for rel in FROZEN:
        p = repo / rel
        st["hashes"][Path(rel).name] = sha256_file(p) if p.is_file() else None
    if st["head"] != BASE:
        st["problems"].append(f"HEAD {st['head']} != {BASE[:10]}")
    if st["changed_paths"] != EXPECTED_PATHS:
        st["problems"].append(f"unexpected working-tree changes: {st['changed_paths']}")
    for rel, want in FROZEN.items():
        if st["hashes"].get(Path(rel).name) != want:
            st["problems"].append(f"hash mismatch {rel}")
    st["ok"] = not st["problems"]
    return st


def save_repo_state(c, at):
    s = repo_state(c.repo)
    jwrite(at / "repo_state_after.json", s)
    return s


def collect_meta(c, at: Path, stage: str, gpu=None, extra=None) -> dict:
    man = verify_manifest()
    meta = dict(package=PACKAGE_NAME, stage=stage, started=now_iso(), smoke=c.smoke, argv=sys.argv, config_file=c.cfg_file,
                config={k: v for k, v in c.cfg.items()}, seeds=dict(cohort_seed=c.cfg["R3_COHORT_SEED"], script_A_seed=c.cfg["R3_A_SEED"], benchmark_order_seed_recorded=20260904),
                host=host_facts(), manifest=dict(ok=man["ok"], manifest_sha256=man["manifest_sha256"], n=man["n"], changed=man["changed"], missing=man["missing"], extra=man["extra"]),
                code_sha256={rel: sha256_file(PKG / rel) for rel in ("r3_scripts/r3_diff_ibi.py", "r3_scripts/r3_global_check_timing.py", "r3_scripts/r3_conn_equiv.py", "p3_frozen/p3_production_integration_gate.py",
                                                                      "lib/cpu_fp.py", "lib/seedref.py", "lib/ibi_cpu.py", "lib/r1_check53.py", "r3tools/r3run.py", "r3tools/r3_eval.py", "r3tools/r3common.py")},
                production=repo_state(c.repo),
                reference_sha256={n: sha256_file(c.ref / n) for n in ("benchmark_metadata.json", "benchmark_rows.csv") if (c.ref / n).is_file()},
                gpu_selected=(gpu or {}).get("gpu") if isinstance(gpu, dict) and "gpu" in gpu else gpu, env={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV")})
    r = subprocess.run([c.py, str(TOOLS / "r3_envprobe.py")], capture_output=True, text=True, env=child_env(c.cfg, None))
    try:
        meta["python_env"] = json.loads(r.stdout)
        meta["cupy_cuda"] = meta["python_env"].get("cupy_cuda")
    except Exception:
        meta["python_env"] = dict(error=(r.stderr or r.stdout)[-400:])
    rc, txt = smi(["-q"], timeout=60)
    if rc == 0:
        (at / "nvidia-smi-q.txt").write_text(txt + "\n", encoding="utf-8")
        meta["nvidia_smi_q_sha256"] = sha256_file(at / "nvidia-smi-q.txt")
        meta["cuda_driver_header"] = next((l.strip() for l in smi([])[1].splitlines() if "CUDA Version" in l), None)
    meta.update(extra or {})
    jwrite(at / "run_meta.json", meta)
    return meta


# ------------------------------------------------------------------------------------------------------------------ GPU helpers
def resolve_gpu(c, rd, stage_no):
    """Stage 0 chooses (first idle GPU if 'auto'); later stages reuse stage 0's GPU after re-checking its identity."""
    kw = dict(min_free_gib=fnum(c.cfg, "R3_MIN_FREE_GPU_GIB"), idle_mem_mib=fnum(c.cfg, "R3_IDLE_MEM_MIB"), idle_util=fnum(c.cfg, "R3_IDLE_UTIL_PCT"))
    if stage_no == 0:
        return pick_gpu(c.cfg["R3_CUDA_DEVICE"], **kw)
    _, rec = find_pass(c, rd, "stage0")
    idx = str(rec.get("gpu_index")) if rec else c.cfg["R3_CUDA_DEVICE"]
    st = gpu_status(idx, **kw)
    if rec and st.get("gpu") and st["gpu"]["uuid"] != rec.get("gpu_uuid"):
        st = dict(status="BUSY", reasons=["GPU identity changed since stage0"], gpu=st["gpu"], foreign=[])
    return st


def make_monitor(c, idx):
    def mon(pid):
        st = gpu_status(idx, own_pids=[pid], min_free_gib=0.0)
        if st["status"] == "NO_GPU":
            return None
        if st["foreign"]:
            return "foreign GPU process appeared: " + "; ".join(f"pid {a['pid']} {a['name']}" for a in st["foreign"]) + " -- stopping OUR job (their job is untouched)"
        return None
    return mon


def skip_step(at, name, why):
    with open(at / "steps.tsv", "a", encoding="utf-8") as f:
        f.write("\t".join([name, "4", now_iso(), now_iso(), "0", "SKIPPED: " + why, "-"]) + "\n")
    say(f"[{name}] SKIPPED: {why}")


# ------------------------------------------------------------------------------------------------------------------ stage 0
def stage0(c):
    rd = run_dir_for(c, create=True)
    at = new_attempt(rd, "stage0")
    say(f"== Stage 0: environment check ({'SMOKE (CPU stand-ins)' if c.smoke else 'real'}) -> {at}")
    chk = []

    def add(cid, ok, detail, required=True):
        chk.append(dict(id=cid, ok=ok, required=required, detail=str(detail)))
        say(f"  [{'OK' if ok else ('--' if not required else 'NG')}] {cid}: {detail}")

    man = verify_manifest()
    add("package.manifest", man["ok"], f"{man['n']} files verified" if man["ok"] else f"changed={man['changed']} missing={man['missing']} err={man.get('error')}")
    if man["extra"]:
        add("package.extra_files", True, f"{len(man['extra'])} unlisted file(s) (ignored): {man['extra'][:3]}", False)
    bad = check_output_location(c.cfg)
    add("output.location", not bad, "; ".join(bad) or f"{c.out_root} (free {disk_free_gib(c.out_root)} GiB)")
    add("output.disk", disk_free_gib(c.out_root) >= fnum(c.cfg, "R3_MIN_DISK_GIB"), f"free {disk_free_gib(c.out_root)} GiB (min {c.cfg['R3_MIN_DISK_GIB']})", not c.smoke)
    hf = host_facts()
    add("host.ram", hf.get("mem_available_gib", 0) >= fnum(c.cfg, "R3_MIN_RAM_GIB") or c.smoke, f"available {hf.get('mem_available_gib')} GiB of {hf.get('mem_total_gib')} (min {c.cfg['R3_MIN_RAM_GIB']}); load {hf.get('loadavg')}", not c.smoke)
    add("host.cpu", True, hf.get("cpu_model", "unknown"), False)
    # python env
    r = subprocess.run([c.py, str(TOOLS / "r3_envprobe.py")], capture_output=True, text=True)
    pe = {}
    try:
        pe = json.loads(r.stdout)
    except Exception:
        pass
    add("python.runs", bool(pe), f"{c.py}" + ("" if pe else f" -> cannot run: {(r.stderr or r.stdout)[-200:]}"))
    mods = pe.get("modules", {})
    for m in ("numpy", "scipy") + (() if c.smoke else ("nibabel", "cupy")):
        add(f"python.{m}", "version" in mods.get(m, {}), mods.get(m, {}).get("version") or mods.get(m, {}).get("error", "missing"))
    add("python.isolated_env", bool(pe.get("isolated_env")) or c.cfg["R3_ALLOW_NON_ISOLATED_PYTHON"] == "1" or c.smoke,
        f"venv/conda active={pe.get('isolated_env')} ({pe.get('executable')}). Create one with `./run.sh setup-venv` or set R3_PYTHON", True)
    rec_env = (jread(c.ref / "benchmark_metadata.json", {}).get("environment") or {})
    add("python.versions_vs_recorded_benchmark", True, f"recorded 4090 run: python {str(rec_env.get('python', ''))[:6]}, numpy {rec_env.get('numpy')}, cupy {rec_env.get('cupy')}, CUDA runtime {rec_env.get('cuda_runtime')}, driver {rec_env.get('driver_version')}; "
        f"this env: python {pe.get('python')}, numpy {mods.get('numpy', {}).get('version')}, cupy {mods.get('cupy', {}).get('version')} (differences are recorded, not blocking)", False)
    # GPU
    gpu = None
    if c.smoke:
        add("gpu.skipped", True, "smoke mode: no GPU is used", False)
    else:
        rc, out = smi(["-L"])
        add("gpu.nvidia_smi", rc == 0, out.replace("\n", " | ") if rc == 0 else f"{out} -> NO GPU: the GPU verification is NOT RUN")
        if rc == 0:
            (at / "nvidia-smi.txt").write_text(smi([])[1] + "\n", encoding="utf-8")
            for g in gpu_list():
                say(f"      GPU {g['index']}: {g['name']} driver {g['driver']} mem {g['mem_used_mib']}/{g['mem_total_mib']} MiB util {g['util_pct']}% {g['uuid'][:16]}")
            apps = []
            from r3common import compute_apps
            apps = compute_apps()
            say(f"      compute processes on this server: {len(apps)}" + ("".join(f"\n        gpu {a['uuid'][:12]} pid {a['pid']} {a['name']} {a['mem_mib']} MiB" for a in apps)))
            gpu = resolve_gpu(c, rd, 0)
            add("gpu.idle", gpu["status"] == "IDLE", f"selected GPU {(gpu.get('gpu') or {}).get('index')} {(gpu.get('gpu') or {}).get('name')}: {gpu['status']} {'; '.join(gpu.get('reasons', []))}. "
                "If another job is running this check FAILS and no GPU experiment is started.")
            cc = pe.get("cupy_cuda") or {}
            add("gpu.cupy_cuda", bool(cc.get("device_count")), f"cupy sees {cc.get('device_count')} device(s); runtime {cc.get('runtime_version')} driver {cc.get('driver_version')}; {cc.get('error', '')}")
    # data / reference / repo
    root = Path(c.cfg["R3_BRATS_ROOT"]).expanduser() if c.cfg["R3_BRATS_ROOT"] else None
    if root and root.is_dir():
        n = len(glob.glob(str(root / "*" / "*_flair.nii*")))
        add("data.brats", n == 1251, f"{n} FLAIR files under {root} (expected 1251)", not c.smoke)
    else:
        add("data.brats", False, f"R3_BRATS_ROOT not set / not a directory ({root}) -- set it in config/r3.env", not c.smoke)
    md = jread(c.ref / "benchmark_metadata.json", {})
    add("data.reference", len(md.get("cohort", [])) == 20 and (c.ref / "benchmark_rows.csv").is_file(), "recorded 4090 benchmark metadata (20-subject cohort) and rows.csv present (read-only copies)")
    rs = repo_state(c.repo)
    add("repo.prepared", rs["ok"], f"{c.repo}: " + ("HEAD 8fdfe62 + frozen P3 patch, 6 production hashes match" if rs["ok"] else "; ".join(rs["problems"]) + "  -> run `./run.sh prepare`"))
    # probe
    required_ok = all(x["ok"] for x in chk if x["required"])
    if c.smoke:
        add("gpu.probe", True, "skipped in smoke mode", False)
    elif required_ok and gpu and gpu["status"] == "IDLE":
        idx = gpu["gpu"]["index"]
        pr = run_step("gpu_probe", [c.py, str(TOOLS / "r3_probe.py"), "--repo", str(c.repo), "--lib", str(PKG / "lib"), "--backend", "cupy", "--out", str(at / "gpu_probe.json")], at,
                      env=child_env(c.cfg, idx), timeout=600, monitor=make_monitor(c, idx), interval=5, echo=False)
        pj = jread(at / "gpu_probe.json", {})
        add("gpu.probe", pr["rc"] == 0 and pj.get("verdict") == "PROBE_PASS", f"12^3 volume: production FP == CPU reference bitwise, global face+cube = 0 violations ({pj.get('gpu')}, {pj.get('gpu_fp_seconds_incl_compile')} s); rc={pr['rc']} {pr['reason']}")
    else:
        add("gpu.probe", False, "not run because an earlier required check failed or the GPU is not idle", True)
    ok = all(x["ok"] for x in chk if x["required"])
    verdict = ("SMOKE_PASS" if c.smoke else "PASS") if ok else "FAIL"
    gsel = (gpu or {}).get("gpu") or {}
    collect_meta(c, at, "stage0", gpu)
    jwrite(at / "verdict.json", dict(stage="stage0", verdict=verdict, failure_kind=None if ok else "ENVIRONMENT", evidence_class="SMOKE_NOT_GPU" if c.smoke else "ENVIRONMENT_ONLY", checks=chk, evaluated=now_iso()))
    (at / "RESULT_SUMMARY.md").write_text(f"# stage0 판정: **{verdict}**\n\n" + "\n".join(f"- [{'OK' if x['ok'] else ('NG' if x['required'] else '--')}] {x['id']}: {x['detail']}" for x in chk) + "\n", encoding="utf-8")
    if ok:
        write_pass(c, at, "stage0", gsel)
    seal(at)
    say("")
    say(f"STAGE0 {verdict}" + ("" if ok else " -- GPU 검증은 실행되지 않았습니다(NOT RUN). 위 [NG] 항목을 해결한 뒤 `./run.sh stage0`을 다시 실행하십시오."))
    if ok and not c.smoke:
        say("다음: ./run.sh stage1")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------------------ prepare / setup-venv
def prepare(c):
    c.out_root.mkdir(parents=True, exist_ok=True)
    logs = c.out_root / f"prepare_{stamp()}"
    if (c.repo / ".git").exists():
        rs = repo_state(c.repo)
        say(f"{c.repo} already exists: " + ("valid (HEAD 8fdfe62 + frozen patch, hashes match). Nothing changed." if rs["ok"] else "INVALID: " + "; ".join(rs["problems"]) + "\nThe runner never modifies an existing checkout. Choose another R3_REPO or move it away."))
        return 0 if rs["ok"] else 1
    if c.repo.exists() and any(c.repo.iterdir()):
        say(f"ERROR: {c.repo} exists and is not empty (not a git checkout). Refusing to write there.")
        return 2
    steps = [("clone_bundle", ["git", "clone", "-n", str(PKG / "production" / "ddfp_base_8fdfe62.bundle"), str(c.repo)]),
             ("checkout_base", ["git", "-C", str(c.repo), "checkout", "--detach", BASE]),
             ("apply_check", ["git", "-C", str(c.repo), "apply", "--check", str(PKG / "production" / "PATCH.diff")]),
             ("apply_patch", ["git", "-C", str(c.repo), "apply", str(PKG / "production" / "PATCH.diff")])]
    for name, argv in steps:
        r = run_step(name, argv, logs)
        if r["rc"] != 0:
            say(f"ERROR: step {name} failed (rc={r['rc']}); see {r['log']}. Partial checkout left at {c.repo} for inspection.")
            return 1
    rs = repo_state(c.repo)
    say("production hash gate: " + ("SERVER_REPO_PRODUCTION_HASH_GATE_PASS" if rs["ok"] else "FAIL " + "; ".join(rs["problems"])))
    r = run_step("p3_preflight", [c.py, str(PKG / "p3_frozen" / "p3_production_integration_gate.py"), "--repo", str(c.repo), "--p2-root", str(PKG / "p3_frozen" / "P2_GATE_INPUTS"), "--preflight-only"], logs, env=child_env(c.cfg), echo=False)
    say(f"P3 preflight (no CUDA): rc={r['rc']}  logs: {logs}")
    return 0 if rs["ok"] and r["rc"] == 0 else 1


def setup_venv(c):
    logs = c.out_root / f"setup_venv_{stamp()}"
    venv = Path(c.cfg["R3_VENV"]).expanduser()
    if venv.exists() and any(venv.iterdir()):
        say(f"{venv} already exists -- not touched. Activate it with: source {venv}/bin/activate  (or set R3_PYTHON={venv}/bin/python)")
        return 0
    pkgs = ["numpy", "scipy", "nibabel", "tqdm"]
    cup = c.args.cupy_package
    if not cup:
        head = next((l for l in smi([])[1].splitlines() if "CUDA Version" in l), "")
        major = head.split("CUDA Version:")[1].strip().split(".")[0] if "CUDA Version:" in head else ""
        cup = {"13": "cupy-cuda13x", "12": "cupy-cuda12x", "11": "cupy-cuda11x"}.get(major, "")
        say(f"nvidia-smi header: '{head.strip()}' -> CuPy wheel: {cup or 'UNKNOWN (pass --cupy-package cupy-cuda12x|cupy-cuda13x)'}")
    if not cup:
        return 2
    cmds = [("venv", [sys.executable, "-m", "venv"] + (["--system-site-packages"] if c.args.system_site_packages else []) + [str(venv)]),
            ("pip_upgrade", [str(venv / "bin" / "python"), "-m", "pip", "install", "--upgrade", "pip"]),
            ("pip_install", [str(venv / "bin" / "python"), "-m", "pip", "install"] + pkgs + [cup])]
    for name, argv in cmds:
        r = run_step(name, argv, logs)
        if r["rc"] != 0:
            say(f"ERROR: {name} failed (rc={r['rc']}); see {r['log']}. (No network / no pip? install the packages by hand into {venv}.)")
            return 1
    say(f"OK. Set R3_PYTHON={venv}/bin/python in config/r3.env (or `source {venv}/bin/activate`), then run ./run.sh stage0")
    return 0


# ------------------------------------------------------------------------------------------------------------------ stages 1-3
def common_gpu_stage(c, stage_name, prev):
    rd = run_dir_for(c, create=False)
    ok, msg, prec = gate_check(c, rd, prev)
    say(f"gate: {msg}")
    if not ok:
        return None
    at = new_attempt(rd, stage_name)
    say(f"== {stage_name} -> {at}")
    gpu = None if c.smoke else resolve_gpu(c, rd, 1)
    return rd, at, prec, gpu


def eval_cfg(c, **extra):
    hf = host_facts()
    d = dict(ref_csv=str(c.ref / "benchmark_rows.csv"), tolerance=fnum(c.cfg, "R3_TIMING_TOLERANCE"), mem_available_gib=hf.get("mem_available_gib"), disk_free_gib=disk_free_gib(c.out_root),
             min_ram_gib=0 if c.smoke else fnum(c.cfg, "R3_MIN_RAM_GIB"), min_disk_gib=0 if c.smoke else fnum(c.cfg, "R3_MIN_DISK_GIB"), cpu_check_n=3, seq_subjects=3)
    d.update(extra)
    return json.dumps(d)


def finish_stage(c, at, stage, rc_eval, gpu, repo_after=True):
    if repo_after:
        save_repo_state(c, at)
    # evaluator is run by caller; here we only open the gate and seal
    return rc_eval


def evaluate(c, at, cmd, **cfg_extra):
    argv = [sys.executable, str(TOOLS / "r3_eval.py"), cmd, "--dir", str(at), "--config-json", eval_cfg(c, **cfg_extra)] + (["--smoke"] if c.smoke else [])
    r = subprocess.run(argv)
    return r.returncode


def stage1(c):
    got = common_gpu_stage(c, "stage1", "stage0")
    if got is None:
        return 3
    rd, at, prec, gpu = got
    py = c.py
    idx = None
    if c.smoke:
        collect_meta(c, at, "stage1", None)
        run_step("a_shim", [py, str(SCRIPTS / "r3_diff_ibi.py"), "--repo", str(c.repo), "--lib", str(PKG / "lib"), "--mode", "shim", "--n-random", "12", "--mutation-test", "--out", str(at / "r3A_shim.json")], at)
    else:
        collect_meta(c, at, "stage1", gpu)
        if gpu["status"] != "IDLE":
            for n in ("p3_gate", "a_gpu_lower"):
                skip_step(at, n, "GPU not idle: " + "; ".join(gpu["reasons"]))
        else:
            idx = gpu["gpu"]["index"]
            env, mon = child_env(c.cfg, idx), make_monitor(c, idx)
            tmo = fnum(c.cfg, "R3_STAGE1_TIMEOUT_S")
            run_step("p3_preflight", [py, str(PKG / "p3_frozen" / "p3_production_integration_gate.py"), "--repo", str(c.repo), "--p2-root", str(PKG / "p3_frozen" / "P2_GATE_INPUTS"), "--preflight-only"], at, env=env, echo=False)
            run_step("p3_gate", [py, str(PKG / "p3_frozen" / "p3_production_integration_gate.py"), "--repo", str(c.repo), "--p2-root", str(PKG / "p3_frozen" / "P2_GATE_INPUTS"), "--out", str(at / "p3_result.json")], at, env=env, timeout=tmo, monitor=mon)
            if not skip_if_busy(c, at, idx, "a_gpu_lower"):
                ra = run_step("a_gpu_lower", [py, str(SCRIPTS / "r3_diff_ibi.py"), "--repo", str(c.repo), "--lib", str(PKG / "lib"), "--mode", "gpu", "--n-random", c.cfg["R3_A_N_RANDOM"], "--seed", c.cfg["R3_A_SEED"],
                                              "--tie", "lower", "--out", str(at / "r3A_gpu_lower.json")], at, env=env, timeout=tmo, monitor=mon)
                if ra["rc"] == 1 and not skip_if_busy(c, at, idx, "a_gpu_higher"):          # diagnostic only: does a tie-policy difference explain the mismatch?
                    run_step("a_gpu_higher", [py, str(SCRIPTS / "r3_diff_ibi.py"), "--repo", str(c.repo), "--lib", str(PKG / "lib"), "--mode", "gpu", "--n-random", c.cfg["R3_A_N_RANDOM"], "--seed", c.cfg["R3_A_SEED"],
                                              "--tie", "higher", "--out", str(at / "r3A_gpu_higher.json")], at, env=env, timeout=tmo, monitor=mon)
    save_repo_state(c, at)
    rc = evaluate(c, at, "stage1")
    gsel = (gpu or {}).get("gpu") if gpu else None
    opened = write_pass(c, at, "stage1", gsel) if rc == 0 else False
    seal(at)
    verdict_msg(c, "STAGE1", rc, opened, "./run.sh stage2")
    return rc


def skip_if_busy(c, at, idx, name):
    st = gpu_status(idx, min_free_gib=fnum(c.cfg, "R3_MIN_FREE_GPU_GIB"), idle_mem_mib=fnum(c.cfg, "R3_IDLE_MEM_MIB"), idle_util=fnum(c.cfg, "R3_IDLE_UTIL_PCT"))
    if st["status"] != "IDLE":
        skip_step(at, name, "GPU not idle before step: " + "; ".join(st["reasons"]))
        return True
    return False


def b_args(c, at, outname, n, cpu_n, seq_n, idx):
    base = [c.py, str(SCRIPTS / "r3_global_check_timing.py"), "--repo", str(c.repo), "--lib", str(PKG / "lib"), "--n", str(n), "--reps", "1", "--seed", c.cfg["R3_COHORT_SEED"], "--max-rounds", "8",
            "--cpu-check-n", str(cpu_n), "--seq-subjects", str(seq_n), "--out", str(at / outname)]
    if c.smoke:
        return base + ["--backend", "stub", "--synthetic", "24"]
    return base + ["--brats-root", c.cfg["R3_BRATS_ROOT"], "--metadata", str(c.ref / "benchmark_metadata.json"), "--ref-csv", str(c.ref / "benchmark_rows.csv"),
                   "--min-free-gib", c.cfg["R3_MIN_FREE_GPU_GIB"], "--gpu-index", str(idx)]


def stage2(c):
    got = common_gpu_stage(c, "stage2", "stage1")
    if got is None:
        return 3
    rd, at, prec, gpu = got
    idx = None if c.smoke else (gpu["gpu"]["index"] if gpu.get("gpu") else None)
    collect_meta(c, at, "stage2", gpu)
    seq_n = 1 if c.cfg["R3_PILOT_SEQ"] == "1" else 0
    if c.smoke:
        run_step("b_pilot", b_args(c, at, "b_pilot", 1, 1, seq_n, None), at, timeout=900)
    elif gpu["status"] != "IDLE":
        skip_step(at, "b_pilot", "GPU not idle: " + "; ".join(gpu["reasons"]))
    else:
        run_step("b_pilot", b_args(c, at, "b_pilot", 1, 1, seq_n, idx), at, env=child_env(c.cfg, idx), timeout=fnum(c.cfg, "R3_PILOT_TIMEOUT_S"), monitor=make_monitor(c, idx))
    save_repo_state(c, at)
    rc = evaluate(c, at, "pilot")
    opened = write_pass(c, at, "stage2", gpu.get("gpu") if gpu else None) if rc == 0 else False
    seal(at)
    verdict_msg(c, "STAGE2 (N=1 pilot)", rc, opened, "./run.sh stage3 --confirm N20   (결과를 먼저 검토한 뒤 실행)")
    return rc


def stage3(c):
    if c.args.confirm != "N20":
        say("Stage 3 (N=20) is never started by default. After reviewing the pilot, run:  ./run.sh stage3 --confirm N20")
        return 3
    got = common_gpu_stage(c, "stage3", "stage2")
    if got is None:
        return 3
    rd, at, prec, gpu = got
    pv = jread(Path(prec["_attempt"]) / "verdict.json", {})
    feas = pv.get("feasibility") or {}
    if not feas.get("feasible"):
        say(f"N=20 not feasible according to the pilot evaluation: {feas.get('summary')}")
        skip_step(at, "b_n20", "pilot feasibility = false")
        n = 0
    else:
        n = 3 if c.smoke else 20
    idx = None if c.smoke else (gpu["gpu"]["index"] if gpu.get("gpu") else None)
    collect_meta(c, at, "stage3", gpu, dict(pilot_attempt=prec["_attempt"], watchdog_s=feas.get("watchdog_s")))
    wd = float(feas.get("watchdog_s") or 1800) if not c.smoke else 900
    if n:
        if c.smoke:
            run_step("b_n20", b_args(c, at, "b_n20", n, 1, 0, None), at, timeout=wd)
        elif gpu["status"] != "IDLE":
            skip_step(at, "b_n20", "GPU not idle: " + "; ".join(gpu["reasons"]))
        else:
            run_step("b_n20", b_args(c, at, "b_n20", n, 3, 3, idx), at, env=child_env(c.cfg, idx), timeout=wd, monitor=make_monitor(c, idx))
    save_repo_state(c, at)
    rc = evaluate(c, at, "n20", n=n or 20)
    opened = write_pass(c, at, "stage3", gpu.get("gpu") if gpu else None) if rc == 0 else False
    seal(at)
    verdict_msg(c, "STAGE3 (N=20)", rc, opened, "결과 폴더 전체를 가져와 새 세션에서 R3 보고서에 반영")
    return rc


def verdict_msg(c, label, rc, opened, nxt):
    v = {0: "PASS" if not c.smoke else "SMOKE_PASS", 1: "FAIL", 2: "INCOMPLETE", 3: "INTERNAL ERROR"}.get(rc, str(rc))
    say("")
    say(f"{label}: {v}" + (f"  -> next: {nxt}" if rc == 0 else "  -> 로그와 중간 결과는 그대로 보존되었습니다. RESULT_SUMMARY.md 를 확인하십시오."))
    if c.smoke:
        say("(SMOKE: CPU 대체 구현 시험입니다. GPU 검증이 아니며 실제 단계의 게이트를 열지 않습니다.)")


def status(c):
    rd = run_dir_for(c, create=False)
    say(f"run: {rd}")
    for st in ("stage0", "stage1", "stage2", "stage3"):
        for at in attempts(rd, st):
            v = jread(at / "verdict.json", {})
            gate = (at / gate_file(c)).is_file()
            say(f"  {st:7s} {at.name:28s} {v.get('verdict', '(no verdict)'):11s} gate={'open' if gate else '-'}  {v.get('evidence_class', '')}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="stage0", choices=["stage0", "prepare", "stage1", "stage2", "stage3", "status", "verify", "setup-venv"])
    ap.add_argument("--config"); ap.add_argument("--smoke", action="store_true", help="CPU stand-ins (self-test); never GPU evidence")
    ap.add_argument("--new-run", action="store_true"); ap.add_argument("--run-id")
    ap.add_argument("--brats-root"); ap.add_argument("--out-root"); ap.add_argument("--repo"); ap.add_argument("--gpu"); ap.add_argument("--python")
    ap.add_argument("--confirm", help="stage3 only: must be N20")
    ap.add_argument("--cupy-package"); ap.add_argument("--system-site-packages", action="store_true")
    a = ap.parse_args()
    try:
        c = make_ctx(a)
        if a.command == "verify":
            m = verify_manifest()
            say(json.dumps(m, indent=1))
            return 0 if m["ok"] else 1
        return dict(stage0=stage0, prepare=prepare, stage1=stage1, stage2=stage2, stage3=stage3, status=status)[a.command](c) if a.command != "setup-venv" else setup_venv(c)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        say("\nINTERRUPTED (Ctrl-C). Only this runner's own child was stopped; logs and intermediate files are kept. Re-run the stage to start a NEW attempt folder.")
        return 130
    except Exception:
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
