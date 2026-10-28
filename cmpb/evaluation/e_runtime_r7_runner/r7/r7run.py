#!/usr/bin/env python3
"""DD-FP R7 server runner -- orchestrator (stdlib only; children use the Python of the isolated environment).  Use through run7.sh.  Rootless: no privilege escalation anywhere.

  preflight   read-only environment / code-hash / GPU-contention check (no CUDA kernel, no allocation).  Compares an existing server tree with the sealed hashes.
  prepare     build the sealed production checkout offline from the bundled git bundle + PATCH.diff (R3 code, unchanged); verifies the 6 hashes.
  setup-venv  create an isolated environment (explicit, optional).
  smoke       seconds-long CUDA plumbing check of E1/E6/E4 scripts on tiny SYNTHETIC volumes (not evidence, opens no gate).  gate: preflight PASS (meant to run after e2 PASS)
  e2          GPU parity 50 fixtures x 2 tie policies = 100 runs on the real GPU (+ CPU self-check).        gate: preflight PASS
  e1          BraTS N=20: GPU K=1 full output vs same-tie CPU state machine, Sequential FP, global DWC.      gate: e2 PASS
  e6          same-server benchmark K=1 / DD-FP K=16 / Sequential FP, 3 repetitions (+ E3 verification costs).  gate: e1 PASS
  e3          cost comparison seam scan vs global check, derived from the e6 rows (no GPU).                   gate: e6 PASS
  e4          safe stepwise memory scaling (never provokes OOM).  needs --confirm E4.                         gate: e6 PASS
  status | verify-pkg | collect | verify-results | selftest
A stage opens only if the previous stage's verdict is PASS (same package hash, same GPU).  After a FAIL the next stage stays closed until the cause is judged:
rerun the failed stage after the fix, or pass --triage-note "<your judgement>" (FAIL only; recorded in the new attempt).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG / "r3tools")); sys.path.insert(0, str(PKG / "r7"))
import r3common as R3          # noqa: E402  (unchanged R3 helper module)
import r3run                   # noqa: E402  (unchanged R3 orchestrator: prepare / setup_venv / repo_state / seal)
import r7_common as C          # noqa: E402

PY_SCRIPTS = PKG / "r7"
STAGES = ["preflight", "e2", "e1", "e6", "e3", "e4"]
GATE = {"e2": "preflight", "e1": "e2", "e6": "e1", "e3": "e6", "e4": "e6"}
say = r3run.say


class Tee:
    def __init__(self, path): self.f, self.o = open(path, "a", encoding="utf-8", buffering=1), sys.stdout
    def write(self, s): self.o.write(s); self.f.write(s)
    def flush(self): self.o.flush(); self.f.flush()
    def isatty(self): return False


def make_ctx(a):
    a.smoke, a.cupy_package, a.system_site_packages, a.new_run = False, getattr(a, "cupy_package", None), getattr(a, "system_site_packages", False), getattr(a, "new_run", False)
    a.config = a.config or str(PKG / "config" / "r7.env")
    c = r3run.make_ctx(a); here = PKG.parent
    # R3 helper defaults point to R3 names; give R7 its own (an explicit config / command-line value is never replaced)
    for key, r3name, r7name in (("R3_OUT_ROOT", "ddfp_r3_runs", "ddfp_r7_runs"), ("R3_REPO", "DD-FP-R3", "DD-FP-R7"), ("R3_VENV", "ddfp_r3_venv", "ddfp_r7_venv")):
        if c.cfg[key] == str(here / r3name): c.cfg[key] = str(here / r7name)
    vp = Path(c.cfg["R3_VENV"]) / "bin" / "python"
    explicit = a.python or (R3.parse_env_file(a.config).get("R3_PYTHON") if Path(a.config).is_file() else "")
    if not explicit and vp.is_file(): c.cfg["R3_PYTHON"] = str(vp)
    c.out_root, c.py, c.repo = Path(c.cfg["R3_OUT_ROOT"]).expanduser(), c.cfg["R3_PYTHON"], Path(c.cfg["R3_REPO"]).expanduser()
    return c


def run_dir(c, create):
    ptr = c.out_root / "CURRENT_R7_RUN"
    if create:
        bad = R3.check_output_location(c.cfg)
        if bad: raise SystemExit("ERROR: " + "; ".join(bad))
        c.out_root.mkdir(parents=True, exist_ok=True)
        rid = c.args.run_id or ((ptr.read_text().strip() if ptr.is_file() and not c.args.new_run else "") or f"r7_{R3.stamp()}")
        ptr.write_text(rid + "\n")
    else:
        rid = c.args.run_id or (ptr.read_text().strip() if ptr.is_file() else "")
        if not rid: raise SystemExit("ERROR: no current R7 run. Run `bash run7.sh preflight` first.")
    d = c.out_root / rid; d.mkdir(parents=True, exist_ok=True); return d


def attempts(rd, stage): return sorted((rd / stage).glob("attempt_*"), reverse=True) if (rd / stage).is_dir() else []


def new_attempt(rd, stage, resume=False, extend=False):
    """resume: continue the latest attempt when it never finished (no verdict) or ended INCOMPLETE.  extend: also continue a finished one (e6 --extend adds repetitions).
    The earlier seal / verdict are kept next to the new ones (renamed, never deleted), so the history stays visible."""
    old = attempts(rd, stage)
    if (resume or extend) and old:
        v = verdict_of(old[0])
        if extend or v in (None, "INCOMPLETE"):
            for fn in ("ARTIFACT_SHA256.txt", "verdict.json", "RESULT_SUMMARY.md"):
                f = old[0] / fn
                if f.is_file(): f.rename(old[0] / f"{f.stem}.before_resume_{R3.stamp()}{f.suffix}")
            say(f"resuming attempt {old[0]} (previous verdict: {v})"); return old[0]
    at = rd / stage / f"attempt_{R3.stamp()}"; n = 0
    while at.exists(): n += 1; at = rd / stage / f"attempt_{R3.stamp()}_{n}"
    at.mkdir(parents=True); return at


def verdict_of(at): return (R3.jread(at / "verdict.json", {}) or {}).get("verdict")


def seal_ok(at):
    """verdict.json must be unchanged since the attempt was sealed."""
    f = at / "ARTIFACT_SHA256.txt"
    if not f.is_file(): return False
    h = {l.split("  ", 1)[1]: l.split("  ", 1)[0] for l in f.read_text().splitlines() if "  " in l}
    return h.get("verdict.json") == R3.sha256_file(at / "verdict.json")


def gate(c, rd, stage, uuid=None):
    """-> (ok, message, prerequisite_attempt, override)"""
    pre = GATE.get(stage)
    if not pre: return True, "no prerequisite", None, None
    man = R3.verify_manifest()["manifest_sha256"]; last = None
    for at in attempts(rd, pre):
        v = verdict_of(at); last = last or (at, v)
        sm = R3.jread(at / "stage_meta.json", {}) or {}
        same_pkg = (sm.get("manifest") or {}).get("manifest_sha256") == man
        same_gpu = (uuid is None) or ((sm.get("gpu_selected") or {}).get("uuid") in (None, uuid))
        if v == "PASS" and seal_ok(at) and same_pkg and same_gpu: return True, f"{pre} PASS ({at.name})", at, None
    if last is None: return False, f"STOP: {pre} has not been run in this run.", None, None
    msg = f"STOP: latest {pre} verdict is {last[1]} ({last[0].name}); no sealed PASS for this package/GPU. The next stage is not started."
    overridable = last[1] == "FAIL" and pre in ("e2", "e1", "e6")        # an environment/code preflight FAIL is never overridable; INVALID/INCOMPLETE are not either
    if overridable and c.args.triage_note:
        return True, f"OVERRIDE after FAIL with triage note", last[0], dict(prerequisite=pre, prerequisite_attempt=last[0].name, prerequisite_verdict="FAIL", note=c.args.triage_note, time=R3.now_iso())
    return False, msg + (" (experiment-stage FAIL only: after judging the cause you may pass --triage-note \"...\")" if overridable else ""), None, None


def gpu_state(c, rd, require_uuid=None, min_free=None):
    kw = dict(min_free_gib=min_free if min_free is not None else R3.fnum(c.cfg, "R3_MIN_FREE_GPU_GIB"), idle_mem_mib=R3.fnum(c.cfg, "R3_IDLE_MEM_MIB"), idle_util=R3.fnum(c.cfg, "R3_IDLE_UTIL_PCT"))
    st = R3.pick_gpu(c.cfg["R3_CUDA_DEVICE"], **kw)
    if require_uuid and st.get("gpu") and st["gpu"]["uuid"] != require_uuid: st = dict(status="BUSY", reasons=["GPU identity changed since preflight"], gpu=st["gpu"], foreign=[])
    return st


def monitor_for(idx):
    def mon(pid):
        st = R3.gpu_status(idx, own_pids=[pid], min_free_gib=0.0)
        if st["status"] != "NO_GPU" and st["foreign"]:
            return "foreign GPU process appeared: " + "; ".join(f"pid {a['pid']} {a['name']}" for a in st["foreign"]) + " -- stopping OUR job only"
    return mon


def stage_meta(c, at, stage, gpu=None, extra=None):
    man = R3.verify_manifest()
    m = dict(package="DDFP_R7_GPU_SERVER_RUNNER_R0", r7_version=C.R7_VERSION, stage=stage, started=R3.now_iso(), argv=sys.argv, config_file=c.cfg_file, config=c.cfg,
             host=R3.host_facts(), manifest=dict(ok=man["ok"], manifest_sha256=man["manifest_sha256"], n=man["n"], changed=man["changed"], missing=man["missing"]),
             production=r3run.repo_state(c.repo), gpu_selected=((gpu or {}).get("gpu")), code_sha256={p.name: R3.sha256_file(p) for p in sorted(PY_SCRIPTS.glob("*.py"))},
             seeds=dict(cohort_seed=c.cfg["R3_COHORT_SEED"], order_seed_recorded_benchmark=20260904, bootstrap_seed=20261014), env={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV")})
    r = subprocess.run([c.py, "-I", str(PKG / "r3tools" / "r3_envprobe.py")], capture_output=True, text=True, env=R3.child_env(c.cfg, None))
    try: m["python_env"] = json.loads(r.stdout)
    except Exception: m["python_env"] = dict(error=(r.stderr or r.stdout)[-400:])
    rc, txt = R3.smi(["-q"], timeout=60)
    if rc == 0: (at / "nvidia-smi-q.txt").write_text(txt + "\n", encoding="utf-8"); m["nvidia_smi_q_sha256"] = R3.sha256_file(at / "nvidia-smi-q.txt")
    m.update(extra or {})
    if (at / "stage_meta.json").is_file(): (at / "stage_meta.json").rename(at / f"stage_meta.before_resume_{R3.stamp()}.json")
    R3.jwrite(at / "stage_meta.json", m); return m


def common_args(c, idx, at, name):
    return ["--repo", str(c.repo), "--lib", str(PKG / "lib"), "--out", str(at / name), "--brats-root", c.cfg["R3_BRATS_ROOT"], "--metadata", str(Path(c.cfg["R3_REFERENCE_DIR"]) / "benchmark_metadata.json"), "--gpu-index", str(idx)]


def verify_stage(c, stage, at, extra=()):
    r = subprocess.run([c.py, "-I", str(PY_SCRIPTS / "r7_verify.py"), stage, "--dir", str(at), "--pkg", str(PKG), *extra], env=R3.child_env(c.cfg, None))
    return r.returncode


def prelude(c, stage, need_gpu=True, resume=False, extend=False):
    rd = run_dir(c, create=False)
    uuid = (R3.jread((attempts(rd, "preflight") or [rd])[0] / "stage_meta.json", {}) or {}).get("gpu_selected", {}) or {}
    uuid = uuid.get("uuid")
    ok, msg, pre, ov = gate(c, rd, stage, uuid if need_gpu else None); say(f"gate: {msg}")
    if not ok: return None
    rs = r3run.repo_state(c.repo)
    if not rs["ok"]: say("STOP: sealed production checkout is not valid: " + "; ".join(rs["problems"]) + "  -> bash run7.sh prepare"); return None
    gpu = None
    if need_gpu:
        gpu = gpu_state(c, rd, uuid)
        if gpu["status"] != "IDLE": say(f"STOP: GPU is not usable ({gpu['status']}): {'; '.join(gpu['reasons'])}. Nothing was started; no other process was touched."); return None
    at = new_attempt(rd, stage, resume, extend)
    if ov: R3.jwrite(at / "gate_override.json", ov)
    return rd, at, gpu, pre


def finish_stage(c, stage, at, rc_step, extra_verify=()):
    verify_stage(c, stage, at, extra_verify); r3run.seal(at); v = verdict_of(at)
    say(""); say(f"{stage.upper()} VERDICT: {v}" + ("  -> next: " + {"e2": "bash run7.sh e1", "e1": "bash run7.sh e6", "e6": "bash run7.sh e3 ; bash run7.sh e4 --confirm E4", "e4": "bash run7.sh collect"}.get(stage, "") if v == "PASS" else
                                                    f"  -> the next stage stays CLOSED. Send the whole folder {at} for judgement; do not retry with changed conditions."))
    return 0 if v == "PASS" else (1 if v == "FAIL" else 2)


# ---------------------------------------------------------------------------------------------------------------------------------- stages
PROBE = ("import sys,time;sys.path[:0]=[sys.argv[1]];import numpy as np,cpu_fp;v=np.random.default_rng(0).integers(0,256,(40,)*3).astype(np.float32);lo,hi,l=cpu_fp.build_ispan_cpu(v);"
         "t=time.perf_counter();cpu_fp.fp_cpu(lo,hi,l);print((time.perf_counter()-t)/79**3*1e6)")        # us per expanded cell of the frozen Sequential FP on this CPU (planning aid only)


def cmd_preflight(c):
    rd = run_dir(c, create=True); at = new_attempt(rd, "preflight"); chk = []
    say(f"== R7-B preflight (read-only) -> {at}")

    def add(cid, ok, detail, required=True):
        chk.append(dict(id=cid, ok=bool(ok), required=required, detail=str(detail))); say(f"  [{'OK' if ok else ('--' if not required else 'NG')}] {cid}: {detail}")
    man = R3.verify_manifest(); add("package.manifest", man["ok"], f"{man['n']} files verified" if man["ok"] else f"changed={man['changed']} missing={man['missing']} err={man.get('error')}")
    bad = R3.check_output_location(c.cfg); add("output.location", not bad, "; ".join(bad) or f"{c.out_root}")
    add("output.disk", R3.disk_free_gib(c.out_root) >= R3.fnum(c.cfg, "R3_MIN_DISK_GIB"), f"free {R3.disk_free_gib(c.out_root)} GiB at {c.out_root} (min {c.cfg['R3_MIN_DISK_GIB']})")
    hf = R3.host_facts(); lines = hf.get("loadavg") or [0]
    add("host.ram", hf.get("mem_available_gib", 0) >= R3.fnum(c.cfg, "R3_MIN_RAM_GIB"), f"available {hf.get('mem_available_gib')} of {hf.get('mem_total_gib')} GiB (min {c.cfg['R3_MIN_RAM_GIB']})")
    add("host.cpu_load", lines[0] <= 0.5 * (hf.get("cpu_count") or 1), f"cpu {hf.get('cpu_model')}; {hf.get('cpu_count')} threads; load1 {lines[0]:.2f} (the Sequential timing in E6 needs a quiet CPU)", False)
    try: (at / "lscpu.txt").write_text(subprocess.run(["lscpu"], capture_output=True, text=True, timeout=20).stdout, encoding="utf-8")
    except Exception as e: (at / "lscpu.txt").write_text(f"lscpu unavailable: {e}\n")
    r = subprocess.run([c.py, "-I", str(PKG / "r3tools" / "r3_envprobe.py")], capture_output=True, text=True); pe = {}
    try: pe = json.loads(r.stdout)
    except Exception: pass
    try:
        us = float(subprocess.run([c.py, "-I", "-c", PROBE, str(PKG / "lib")], capture_output=True, text=True, timeout=180).stdout.strip().splitlines()[-1]); cpu_probe = dict(us_per_cell=us, e1_cpu_hours=round(20 * 3 * 71e6 * us * 1e-6 / 3600, 1), e6_sequential_hours=round(60 * 71e6 * us * 1e-6 / 3600, 1))
        add("cpu.sequential_speed_probe", True, f"frozen Sequential FP: {us:.2f} us/expanded cell (40^3 random volume) -> planning estimate for the 71 M-cell recorded mean: one Sequential run ~{71e6 * us * 1e-6:.0f} s; E1 CPU part (3 runs x 20 subjects) ~{cpu_probe['e1_cpu_hours']} h; E6 Sequential (60 runs) ~{cpu_probe['e6_sequential_hours']} h. The recorded benchmark mean was 111.5 s on an unknown CPU. Not a measurement of the real volumes.", False)
    except Exception as e: cpu_probe = dict(error=f"{type(e).__name__}: {e}"); add("cpu.sequential_speed_probe", False, f"probe not available ({type(e).__name__}); needs numpy in the stage python", False)
    mods = pe.get("modules", {}); add("python.runs", bool(pe), f"{c.py} (probed with -I like every stage child: packages installed with `pip --user` are NOT visible; use a venv/conda env)" + ("" if pe else f" -> cannot run: {(r.stderr or r.stdout)[-200:]}"))
    for m in ("numpy", "scipy", "nibabel", "cupy"): add(f"python.{m}", "version" in mods.get(m, {}), mods.get(m, {}).get("version") or mods.get(m, {}).get("error", "missing"))
    add("python.isolated_env", bool(pe.get("isolated_env")) or c.cfg["R3_ALLOW_NON_ISOLATED_PYTHON"] == "1", f"venv/conda active={pe.get('isolated_env')} ({pe.get('executable')})")
    rc, out = R3.smi(["-L"]); add("gpu.nvidia_smi", rc == 0, out.replace("\n", " | ") if rc == 0 else f"{out} -> without a GPU E2/E1/E6/E4 cannot run")
    gsel = {}
    if rc == 0:
        (at / "nvidia-smi.txt").write_text(R3.smi([])[1] + "\n", encoding="utf-8")
        apps = R3.compute_apps(); owners = []
        for a_ in apps:
            try: u = subprocess.run(["ps", "-o", "user=", "-p", a_["pid"]], capture_output=True, text=True, timeout=5).stdout.strip()
            except Exception: u = "?"
            owners.append(dict(a_, user=u))
        say(f"      compute processes on this server: {len(apps)}" + "".join(f"\n        gpu {a_['uuid'][:12]} pid {a_['pid']} user {a_['user']} {a_['name']} {a_['mem_mib']} MiB" for a_ in owners))
        R3.jwrite(at / "gpu_compute_apps.json", owners)
        for g in R3.gpu_list(): say(f"      GPU {g['index']}: {g['name']} driver {g['driver']} mem {g['mem_used_mib']}/{g['mem_total_mib']} MiB util {g['util_pct']}% {g['uuid'][:16]}")
        st = gpu_state(c, rd); gsel = st.get("gpu") or {}
        add("gpu.idle_and_free", st["status"] == "IDLE", f"selected GPU {gsel.get('index')} {gsel.get('name')}: {st['status']} {'; '.join(st.get('reasons', []))}. Another job on it makes this check FAIL and no GPU stage starts.")
        cc = pe.get("cupy_cuda") or {}; add("gpu.cupy_sees_device", bool(cc.get("device_count")), f"cupy devices {cc.get('device_count')}, runtime {cc.get('runtime_version')}, driver {cc.get('driver_version')} {cc.get('error', '')}")
    root = Path(c.cfg["R3_BRATS_ROOT"]).expanduser() if c.cfg["R3_BRATS_ROOT"] else None
    if root and root.is_dir():
        n = len(glob.glob(str(root / "*" / "*_flair.nii*"))); add("data.brats_count", n == 1251, f"{n} FLAIR files under {root} (expected 1251)")
        try:
            sys.path.insert(0, str(PKG / "r7")); cl, _ = C.cohort(root, Path(c.cfg["R3_REFERENCE_DIR"]) / "benchmark_metadata.json", 20, int(c.cfg["R3_COHORT_SEED"])); add("data.cohort_hashes", True, f"recorded 20-subject cohort found and SHA-256 verified ({cl[0][0]} ... {cl[-1][0]})")
        except BaseException as e: add("data.cohort_hashes", False, f"{type(e).__name__}: {e}")
    else: add("data.brats_count", False, f"R3_BRATS_ROOT not set / not a directory ({root})")
    rs = r3run.repo_state(c.repo); add("repo.sealed_checkout", rs["ok"], f"{c.repo}: " + ("HEAD 8fdfe62 + frozen patch, 6 hashes match" if rs["ok"] else "; ".join(rs["problems"]) + "  -> bash run7.sh prepare"), False)
    # ---- the user's existing server tree versus the sealed baseline ----
    st_tree = dict(path=None, compared=False)
    if c.args.server_tree:
        t = Path(c.args.server_tree).expanduser(); head = r3run.git(t, "rev-parse", "HEAD"); status = r3run.git(t, "status", "--porcelain=v1", "--untracked-files=all")
        hs = {k: (R3.sha256_file(t / k) if (t / k).is_file() else None) for k in C.EXPECTED}
        same = head[0] == 0 and head[1] == r3run.BASE and hs == C.EXPECTED
        st_tree = dict(path=str(t), compared=True, head=head[1], status=status[1].splitlines(), hashes=hs, hashes_equal_sealed=hs == C.EXPECTED, head_is_base=head[1] == r3run.BASE)
        add("server_tree.equals_sealed_baseline", same, f"{t}: HEAD {head[1][:10]}; hashes {'equal' if hs == C.EXPECTED else 'DIFFER: ' + str([k for k in C.EXPECTED if hs[k] != C.EXPECTED[k]])}" + ("" if same else "  -> STOP: the server code differs from the sealed baseline; no experiment is started"))
    elif c.args.no_server_tree: add("server_tree.equals_sealed_baseline", True, "NOT COMPARED (--no-server-tree): experiments use only the sealed checkout built from the bundle", False)
    else: add("server_tree.equals_sealed_baseline", False, "pass --server-tree <path of your existing DD-FP tree> to compare it with the sealed hashes, or --no-server-tree to skip knowingly")
    R3.jwrite(at / "server_paths.json", dict(cpu_probe=cpu_probe, package=str(PKG), config=c.cfg_file, brats_root=str(root), out_root=str(c.out_root), sealed_checkout=str(c.repo), python=c.py, venv=c.cfg["R3_VENV"], server_tree=st_tree,
                                              cuda_device_setting=c.cfg["R3_CUDA_DEVICE"], gpu_selected=gsel, host=hf))
    ok = all(x["ok"] for x in chk if x["required"]); v = "PASS" if ok else "FAIL"
    stage_meta(c, at, "preflight", dict(gpu=gsel) if gsel else None)
    R3.jwrite(at / "verdict.json", dict(stage="preflight", verdict=v, checks=chk, evaluated=R3.now_iso(), failure_kind=None if ok else "ENVIRONMENT/CODE", evidence_class="ENVIRONMENT_ONLY"))
    (at / "RESULT_SUMMARY.md").write_text(f"# preflight: **{v}**\n\n" + "\n".join(f"- [{'OK' if x['ok'] else ('NG' if x['required'] else '--')}] {x['id']}: {x['detail']}" for x in chk) + "\n", encoding="utf-8")
    r3run.seal(at); say(""); say(f"PREFLIGHT {v}" + ("  -> next: bash run7.sh prepare (if the sealed checkout is missing), then bash run7.sh e2" if ok else "  -> resolve the [NG] items and run preflight again. No experiment is started."))
    return 0 if ok else 1


def run_child(c, at, name, argv, idx, timeout):
    return R3.run_step(name, argv, at, env=R3.child_env(c.cfg, idx), timeout=timeout, monitor=monitor_for(idx), interval=10)


def cmd_e2(c):
    pr = prelude(c, "e2")
    if not pr: return 2
    rd, at, gpu, _ = pr; idx = gpu["gpu"]["index"]; gp = PKG / "e2" / "gpu_parity" / "run_gpu_parity.py"; stage_meta(c, at, "e2", gpu)
    say(f"== E2: GPU parity 50 x 2 on GPU {idx} {gpu['gpu']['name']} -> {at}")
    r1 = R3.run_step("selfcheck_cpu", [c.py, "-I", str(gp), "--selfcheck", "--no-pause", "--out", str(at / "selfcheck")], at, env=R3.child_env(c.cfg, None), timeout=900)
    r2 = run_child(c, at, "parity_gpu", [c.py, "-I", str(gp), "--no-pause", "--out", str(at / "parity_gpu")], idx, int(c.args.timeout_h * 3600) if c.args.timeout_h else 3600)
    say(f"(the script prints a 3080 Ti hint on other GPUs -- harmless) steps rc: selfcheck={r1['rc']} gpu={r2['rc']} {r2['reason']}")
    return finish_stage(c, "e2", at, r2["rc"])


def cmd_smoke(c):
    """seconds-long CUDA plumbing check of the E1/E6/E4 scripts on tiny SYNTHETIC volumes.  Not evidence of anything, never gates, never replaces a stage."""
    rd = run_dir(c, create=False); pf = (attempts(rd, "preflight") or [rd])[0]; gsel = (R3.jread(pf / "stage_meta.json", {}) or {}).get("gpu_selected") or {}
    ok, msg, _, _ = gate(c, rd, "e2", gsel.get("uuid")); say(f"gate: {msg}")
    if not ok: return 2
    rs = r3run.repo_state(c.repo)
    if not rs["ok"]: say("STOP: sealed production checkout is not valid -> bash run7.sh prepare"); return 2
    gpu = gpu_state(c, rd, gsel.get("uuid"))
    if gpu["status"] != "IDLE": say(f"STOP: GPU is not usable ({gpu['status']}): {'; '.join(gpu['reasons'])}. Nothing was started."); return 2
    at = new_attempt(rd, "smoke"); idx = gpu["gpu"]["index"]; stage_meta(c, at, "smoke", gpu); base = ["--repo", str(c.repo), "--lib", str(PKG / "lib"), "--gpu-index", str(idx)]
    plan = [("e1", ["--synthetic", "2", "--n", "2", "--synthetic-size", "8", "--cpu-check-n", "2"]), ("e6", ["--synthetic", "2", "--n", "2", "--synthetic-size", "8", "--reps", "1", "--max-load1", "1000"]),
            ("e4", ["--synthetic", "1", "--synthetic-size", "8", "--copies", "1,2"])]
    res = {}
    say(f"== SMOKE on GPU {idx} {gpu['gpu']['name']} (tiny synthetic volumes; plumbing only) -> {at}")
    for name, extra in plan:
        r = run_child(c, at, f"smoke_{name}", [c.py, "-I", str(PY_SCRIPTS / f"r7_{name}.py")] + base + ["--out", str(at / name)] + extra, idx, 900); res[name] = dict(rc=r["rc"], reason=r["reason"])
        say(f"  smoke {name}: rc={r['rc']} {r['reason']}")
    sj = sorted((at / "e1" / "subjects").glob("*.json")) if (at / "e1" / "subjects").is_dir() else []
    ps = [json.loads(p.read_text()) for p in sj]; res["e1_parity_ok"] = [x["comparisons"]["parity_lower"]["parity_ok"] and x["comparisons"]["parity_higher"]["parity_ok"] for x in ps]
    res["e1_dwc_ok"] = [all(d["violations"] == 0 for d in x["dwc"].values()) for x in ps]; good = all(res[n]["rc"] == 0 for n, _ in plan)
    v = "SMOKE_RAN" if good else "SMOKE_FAILED"
    R3.jwrite(at / "verdict.json", dict(stage="smoke", verdict=v, results=res, note="plumbing only: tiny synthetic volumes; not a stage, not evidence, never opens a gate", evaluated=R3.now_iso()))
    (at / "RESULT_SUMMARY.md").write_text(f"# smoke: **{v}**\n\n```\n{json.dumps(res, indent=1)}\n```\nPlumbing check on tiny synthetic volumes. It is not E1/E6/E4 and opens no gate. A parity/DWC value above is informational only.\n", encoding="utf-8")
    r3run.seal(at); say(f"SMOKE: {v}" + ("  -> next: bash run7.sh e1" if good else f"  -> do NOT start the real stages; send {at}")); return 0 if good else 1


def cmd_e1(c):
    pr = prelude(c, "e1", resume=c.args.resume)
    if not pr: return 2
    rd, at, gpu, _ = pr; idx = gpu["gpu"]["index"]; stage_meta(c, at, "e1", gpu)
    say(f"== E1: BraTS N={c.args.n} on GPU {idx} {gpu['gpu']['name']} -> {at}  (about 1-2 h; resumable with --resume; Ctrl-C keeps all finished subjects)")
    argv = [c.py, "-I", str(PY_SCRIPTS / "r7_e1.py")] + common_args(c, idx, at, "e1") + ["--n", str(c.args.n), "--cpu-check-n", str(c.args.cpu_check_n)]
    r = run_child(c, at, "e1", argv, idx, int(c.args.timeout_h * 3600) if c.args.timeout_h else 6 * 3600)
    say(f"e1 rc={r['rc']} {r['reason']}")
    return finish_stage(c, "e1", at, r["rc"], ["--expect-n", str(c.args.n)])


def cmd_e6(c):
    pr = prelude(c, "e6", resume=c.args.resume, extend=c.args.extend)
    if not pr: return 2
    rd, at, gpu, pre = pr; idx = gpu["gpu"]["index"]; stage_meta(c, at, "e6", gpu)
    hf = R3.host_facts(); lim = 0.5 * (hf.get("cpu_count") or 1)
    if (hf.get("loadavg") or [0])[0] > lim and not c.args.allow_busy_cpu:
        say(f"STOP: 1-min CPU load {hf['loadavg'][0]:.2f} > {lim:.1f}: other work would contaminate the Sequential timing. Wait or pass --allow-busy-cpu knowingly."); return 2
    say(f"== E6: {c.args.reps} repetitions (start {c.args.reps_start}) on GPU {idx} {gpu['gpu']['name']} -> {at}  (about 2.5-3 h; resumable with --resume)")
    argv = [c.py, "-I", str(PY_SCRIPTS / "r7_e6.py")] + common_args(c, idx, at, "e6") + ["--reps", str(c.args.reps), "--reps-start", str(c.args.reps_start), "--n", str(c.args.n)]
    r = run_child(c, at, "e6", argv, idx, int(c.args.timeout_h * 3600) if c.args.timeout_h else 10 * 3600)
    say(f"e6 rc={r['rc']} {r['reason']}")
    return finish_stage(c, "e6", at, r["rc"], ["--expect-n", str(c.args.n), "--min-reps", "3", "--e1-dir", str(pre)] if pre else ["--expect-n", str(c.args.n)])


def cmd_e3(c):
    rd = run_dir(c, create=False); ok, msg, pre, ov = gate(c, rd, "e3"); say(f"gate: {msg}")
    if not ok: return 2
    at = new_attempt(rd, "e3"); (at / "source_e6_attempt.txt").write_text(f"{pre}\n{R3.sha256_file(pre / 'verdict.json')}\n")
    stage_meta(c, at, "e3", None, dict(source=str(pre)))
    r = subprocess.run([c.py, "-I", str(PY_SCRIPTS / "r7_verify.py"), "e3", "--dir", str(pre), "--out", str(at), "--pkg", str(PKG)], env=R3.child_env(c.cfg, None))
    r3run.seal(at); v = verdict_of(at); say(f"E3 VERDICT: {v}"); return 0 if v == "PASS" else 2


def cmd_e4(c):
    if c.args.confirm != "E4": say("E4 is never started by default. It needs exclusive use of the GPU for up to ~1 h. Check `bash run7.sh preflight`, then: bash run7.sh e4 --confirm E4"); return 2
    pr = prelude(c, "e4")
    if not pr: return 2
    rd, at, gpu, _ = pr; idx = gpu["gpu"]["index"]; g = gpu["gpu"]; stage_meta(c, at, "e4", gpu); hf = R3.host_facts()
    say(f"E4 resource check: GPU {idx} {g['name']} used {gpu['mem_used_mib']:.0f} MiB, free {gpu['free_gib']} GiB, foreign processes {len(gpu['foreign'])}; RAM available {hf.get('mem_available_gib')} GiB; load {hf.get('loadavg')}")
    if gpu["free_gib"] < c.args.e4_min_free_gib: say(f"STOP: only {gpu['free_gib']} GiB free (< {c.args.e4_min_free_gib}). Free the GPU first; E4 is not started."); return 2
    say(f"== E4: safe stepwise scaling (no OOM is provoked) -> {at}")
    argv = [c.py, "-I", str(PY_SCRIPTS / "r7_e4.py")] + common_args(c, idx, at, "e4") + ["--copies", c.args.copies, "--safety-fraction", str(c.args.safety_fraction)]
    r = run_child(c, at, "e4", argv, idx, int(c.args.timeout_h * 3600) if c.args.timeout_h else 3 * 3600)
    say(f"e4 rc={r['rc']} {r['reason']}"); return finish_stage(c, "e4", at, r["rc"])


def cmd_status(c):
    rd = run_dir(c, create=False); say(f"run: {rd}")
    for s in STAGES:
        for at in attempts(rd, s)[:3]: say(f"  {s:<9} {at.name:<28} {verdict_of(at) or '(no verdict yet)'}{'  [sealed]' if seal_ok(at) else ''}")
    return 0


def cmd_collect(c):
    rd = run_dir(c, create=False); out = c.out_root / f"R7_RESULTS_{rd.name}_{R3.stamp()}.zip"; skipped = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(rd.rglob("*")):
            if p.is_file():
                if p.stat().st_size > 150 * 2**20: skipped.append(f"{p.relative_to(rd)} ({p.stat().st_size >> 20} MiB)"); continue
                z.write(p, f"{rd.name}/{p.relative_to(rd).as_posix()}")
        z.writestr(f"{rd.name}/COLLECT_NOTES.txt", f"collected {R3.now_iso()}\nskipped (>150 MiB, keep on the server and send on request):\n" + "\n".join(skipped) + "\n")
    say(f"collected: {out}  ({out.stat().st_size >> 20} MiB)  sha256 {R3.sha256_file(out)}"); (out.with_suffix(".zip.sha256")).write_text(f"{R3.sha256_file(out)}  {out.name}\n"); return 0


def cmd_verify_results(c):
    d = Path(c.args.results).expanduser(); rows = []
    if d.suffix == ".zip":
        import tempfile
        t = Path(tempfile.mkdtemp(prefix="r7res_")); zipfile.ZipFile(d).extractall(t); d = next(t.iterdir())
    for s in ("e2", "e1", "e6", "e4"):
        for at in attempts(d, s)[:1]:
            rc = subprocess.run([c.py, "-I", str(PY_SCRIPTS / "r7_verify.py"), s, "--dir", str(at), "--pkg", str(PKG), "--out", str(at / "reverify")] + (["--e1-dir", str(attempts(d, "e1")[0])] if s == "e6" and attempts(d, "e1") else []), capture_output=True, text=True)
            rows.append((s, at.name, (R3.jread(at / "reverify" / "verdict.json", {}) or {}).get("verdict"), verdict_of(at)))
    for r in rows: say(f"{r[0]:<4} {r[1]:<28} re-verified={r[2]} recorded={r[3]}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["preflight", "prepare", "setup-venv", "smoke", "e2", "e1", "e6", "e3", "e4", "status", "verify-pkg", "collect", "verify-results", "selftest"])
    ap.add_argument("--config"); ap.add_argument("--run-id"); ap.add_argument("--new-run", action="store_true")
    ap.add_argument("--brats-root"); ap.add_argument("--out-root"); ap.add_argument("--repo"); ap.add_argument("--gpu"); ap.add_argument("--python")
    ap.add_argument("--server-tree"); ap.add_argument("--no-server-tree", action="store_true")
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--cpu-check-n", type=int, default=3); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--reps-start", type=int, default=0)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--extend", action="store_true", help="e6 only: add repetitions to the latest e6 attempt (use with --reps-start / --reps)"); ap.add_argument("--timeout-h", type=float); ap.add_argument("--allow-busy-cpu", action="store_true")
    ap.add_argument("--triage-note"); ap.add_argument("--confirm"); ap.add_argument("--copies", default="1,2,3,4,6,8"); ap.add_argument("--safety-fraction", type=float, default=0.75); ap.add_argument("--e4-min-free-gib", type=float, default=16)
    ap.add_argument("--cupy-package"); ap.add_argument("--system-site-packages", action="store_true"); ap.add_argument("--results")
    a = ap.parse_args()
    try:
        if a.command == "verify-pkg": m = R3.verify_manifest(); say(json.dumps(m, indent=1)); return 0 if m["ok"] else 1
        if a.command == "selftest": return subprocess.call([sys.executable, "-B", "-I", str(PKG / "r7" / "r7_selftest.py")])
        c = make_ctx(a)
        if a.command not in ("preflight", "prepare", "setup-venv", "verify-results"):
            run_dir(c, create=False)
        log = (run_dir(c, create=a.command == "preflight") if a.command not in ("prepare", "setup-venv", "collect", "verify-results", "status") else c.out_root); log.mkdir(parents=True, exist_ok=True)
        sys.stdout = Tee(log / f"console_{a.command}_{R3.stamp()}.log")
        fn = dict(preflight=cmd_preflight, prepare=r3run.prepare, smoke=cmd_smoke, e2=cmd_e2, e1=cmd_e1, e6=cmd_e6, e3=cmd_e3, e4=cmd_e4, status=cmd_status, collect=cmd_collect)
        if a.command == "setup-venv": return r3run.setup_venv(c)
        if a.command == "verify-results": return cmd_verify_results(c) if (setattr(c.args, "results", a.results) or True) else 0
        return fn[a.command](c)
    except SystemExit: raise
    except KeyboardInterrupt:
        say("\nINTERRUPTED (Ctrl-C). Only this runner's own child was stopped; logs and finished results are kept. Use --resume for e1/e6."); return 130
    except Exception:
        import traceback; traceback.print_exc(); return 3


if __name__ == "__main__":
    sys.exit(main())
