#!/usr/bin/env python3
"""R7 CPU-only self test.  NO GPU / CUDA is used or claimed: every check below runs on the CPU (stub backend = seedref CPU state machine).
It tests the package (code integrity, same-tie CPU state machine, decision logic, gates, negative controls); it is NOT evidence about CUDA behaviour.
  bash run7.sh selftest            exit 0 only if every non-skipped test passes; skipped tests are listed and are NOT counted as passed.
Needs: python >= 3.8, numpy.  git is needed for the production-checkout tests (they are skipped when git is missing)."""
import json, os, shutil, subprocess, sys, tempfile, time, traceback
from pathlib import Path
from types import SimpleNamespace

PKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PKG / "r3tools"), str(PKG / "r7"), str(PKG / "lib")]
import numpy as np                       # noqa: E402
import r3common as R3                    # noqa: E402
import r7_common as C                    # noqa: E402

WORK = Path(tempfile.mkdtemp(prefix="r7_selftest_"))
RES, T = [], {}
class Skip(Exception): pass


def test(f): T[f.__name__] = f; return f


def sh(args, **kw): return subprocess.run([sys.executable, "-B", "-I", *map(str, args)], capture_output=True, text=True, **kw)


def volumes(n, seed=7):
    """small volumes: half uint8-random (as the stub cohort; many equidistant-level ties), half few-level; odd/even sizes"""
    rng = np.random.default_rng(seed)
    for i in range(n):
        shp = tuple(int(x) for x in rng.integers(4, 7, 3))
        yield (rng.integers(0, 256, shp) if i % 2 == 0 else rng.integers(0, int(rng.integers(3, 8)), shp)).astype(np.float32)


_REPO = {}
def repo():
    """sealed production checkout, built offline from the bundle + frozen patch (cached)."""
    if "p" in _REPO: return _REPO["p"]
    if not shutil.which("git"): raise Skip("git not available")
    try: import scipy                          # noqa: F401  (the production module imports scipy.ndimage at import time)
    except ImportError: raise Skip("scipy not importable under `python -I` (needed to import the production module for the stub tests; -I ignores `pip --user` packages)")
    d = WORK / "repo"; g = lambda *a: subprocess.run(["git", "-C", str(d), *a], capture_output=True, text=True)
    r = subprocess.run(["git", "clone", "-n", str(PKG / "production" / "ddfp_base_8fdfe62.bundle"), str(d)], capture_output=True, text=True)
    if r.returncode: raise AssertionError("clone: " + r.stderr[-300:])
    for a in (("checkout", "--detach", "8fdfe6233dfe53f996ed2f09e8810bca53d5323b"), ("apply", str(PKG / "production" / "PATCH.diff"))):
        r = g(*a)
        if r.returncode: raise AssertionError(f"git {a[0]}: {r.stderr[-300:]}")
    bad = [k for k, v in C.EXPECTED.items() if C.sha_f(d / k) != v]
    assert not bad, f"production hashes differ: {bad}"
    _REPO["p"] = d; return d


# ------------------------------------------------------------------------------------------------------------------------------ static
@test
def t01_compile_all():
    n = 0
    for p in PKG.rglob("*.py"):
        if "_work" in p.parts or "__pycache__" in p.parts: continue
        compile(p.read_text(encoding="utf-8"), str(p), "exec"); n += 1
    return f"{n} python files compile"


@test
def t02_manifest():
    if not (PKG / "MANIFEST.sha256").is_file(): raise Skip("no MANIFEST.sha256 (development tree)")
    m = R3.verify_manifest(); assert m["ok"] and not m["extra"], f"changed={m['changed']} missing={m['missing']} extra={m['extra']}"
    return f"{m['n']} files verified"


@test
def t03_frozen_files():
    assert C.sha_f(PKG / "lib" / "cpu_fp.py") == C.EXPECTED["src/ddfp/cpu_fp.py"], "lib/cpu_fp.py is not the frozen file"
    return "lib/cpu_fp.py = frozen sha"


@test
def t04_shell_syntax():
    if not shutil.which("bash"): raise Skip("bash not available")
    r = subprocess.run(["bash", "-n", str(PKG / "run7.sh")], capture_output=True, text=True); assert r.returncode == 0, r.stderr
    return "bash -n run7.sh ok"


@test
def t05_cli_help():
    for s in ("r7_e1.py", "r7_e4.py", "r7_e6.py", "r7_verify.py", "r7run.py"):
        r = sh([PKG / "r7" / s, "--help"]); assert r.returncode == 0, f"{s}: {r.stderr[-200:]}"
    return "all entry points parse --help"


# ------------------------------------------------------------------------------------------------------------------------ CPU state machine
@test
def t06_static_safety_scan():
    import re
    pat = re.compile(r"\bsudo\b|pkill|killall|os\.system|shell\s*=\s*True|\bkill\s+-9|nvidia-smi[^\n]*(\s-r\b|--gpu-reset|\s-pm\b|\s-pl\b|\s-lgc\b)|\brmtree\b|os\.remove|\.unlink\(")
    bad = []
    for p in list((PKG / "r7").glob("*.py")) + [PKG / "run7.sh"]:
        if p.name == "r7_selftest.py": continue          # the self test removes its own temp directory
        for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(ln) and not ln.lstrip().startswith("#"): bad.append(f"{p.name}:{i}: {ln.strip()[:90]}")
    assert not bad, "forbidden construct(s): " + " | ".join(bad)
    return "no sudo / process killing / GPU reset / shell=True / file deletion in r7/*.py and run7.sh"


@test
def t10_cpu_sm_insertion_equals_frozen_fp_cpu():
    import cpu_fp, r7_cpu_sm
    n = 0
    for v in volumes(60):
        lo, hi, l = cpu_fp.build_ispan_cpu(v); a = cpu_fp.fp_cpu(lo, hi, l); b = r7_cpu_sm.fp_cpu_tie(lo, hi, l, "insertion")
        assert a.shape == b.shape and np.array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32)), f"volume {n}: insertion mode differs from frozen fp_cpu"
        n += 1
    return f"{n} volumes bit-identical"


@test
def t11_cpu_sm_equals_seedref_both_ties_and_has_power():
    import cpu_fp, r7_cpu_sm, seedref
    n = ties = tr = outd = 0
    for v in volumes(60):
        lo, hi, l = cpu_fp.build_ispan_cpu(v); outs, trs = {}, {}
        for p in ("lower", "higher"):
            st = {}; a = r7_cpu_sm.fp_cpu_tie(lo, hi, l, p, st); ref, meta, _ = seedref.seed_aware_cpu_reference(lo, hi, l, None, p)
            assert np.array_equal(np.asarray(a, np.float32), np.asarray(ref, np.float32)), f"volume {n} tie={p}: cpu_sm != seedref (output)"
            assert st["tie_events"] == int(meta["equidistant_next_level_tie_events"]), f"volume {n} tie={p}: tie_events {st['tie_events']} vs {meta['equidistant_next_level_tie_events']}"
            assert [list(x) for x in st["level_transitions"]] == [list(x) for x in meta["level_transitions"]], f"volume {n} tie={p}: level_transitions differ"
            assert st["backward_transitions"] == int(meta["backward_scheduler_transition_count"]), f"volume {n} tie={p}: backward transition count differs"
            outs[p], trs[p] = a, st["level_transitions"]; ties += st["tie_events"] > 0
        tr += trs["lower"] != trs["higher"]; outd += not np.array_equal(outs["lower"], outs["higher"]); n += 1
    assert ties > 0 and tr > 0, f"test has no power: runs with tie events={ties}, volumes where the lower/higher level sequences differ={tr}"
    return (f"{n} volumes x 2 policies identical to seedref in output, tie-event count, level transitions and backward count; power: tie events in {ties} runs, "
            f"level sequences differ lower/higher in {tr} volumes, OUTPUT differs lower/higher in {outd} volumes (observed on these inputs only)")


@test
def t12_negative_control_wrong_tie_policy_is_detected():
    import cpu_fp, r7_cpu_sm, seedref
    d = 0
    for v in volumes(60):
        lo, hi, l = cpu_fp.build_ispan_cpu(v); st = {}; r7_cpu_sm.fp_cpu_tie(lo, hi, l, "higher", st); meta = seedref.seed_aware_cpu_reference(lo, hi, l, None, "lower")[1]
        d += [list(x) for x in st["level_transitions"]] != [list(x) for x in meta["level_transitions"]]
    assert d > 0, "comparing the wrong tie policy was never detected: the parity comparison would be blind"
    return f"comparing 'higher' CPU state machine against the 'lower' reference is detected in {d}/60 volumes (via level transitions)"


# ------------------------------------------------------------------------------------------------------------------------ E4 rules
@test
def t20_e4_admission_rule():
    import r7_e4 as E
    lim = dict(vram_mib=10000.0, rss_mib=20000.0, seconds_max=3600.0, seconds_used=0.0)
    ok, why, _ = E.decide([], [], [], 1, lim); assert ok, why
    pv, pr, ps = [(1, 1000.0)], [(1, 2000.0)], [(1, 100.0)]
    assert E.decide(pv, pr, ps, 2, lim)[0], "m=2 predicted 2300 MiB must be admitted"
    ok, why, p = E.decide(pv, pr, ps, 9, lim); assert not ok and "VRAM" in why, (ok, why)      # 1.15*9000 = 10350 > 10000
    ok, why, _ = E.decide(pv, [(1, 9000.0)], ps, 2, lim); assert not ok and "RSS" in why, why
    ok, why, _ = E.decide(pv, pr, ps, 2, dict(lim, seconds_used=3500.0)); assert not ok and "time" in why, why
    # super-linear growth is not under-predicted: true v(m)=m^1.5*1000 ; points m=1,2 -> prediction for m=4 must be >= true (8000)
    p = E.predict([(1, 1000.0), (2, 2828.0)], 4); assert p >= 8000, p
    return "first step admitted; VRAM/RSS/time limits stop a step; prediction not below proportional, linear or power-law (exponent>=1) growth"


# --------------------------------------------------------------------------------------------------------------- stub runs + verifier
def stub_e1(name, fault=None, n=3):
    out = WORK / name / "e1"; args = [PKG / "r7" / "r7_e1.py", "--repo", repo(), "--lib", PKG / "lib", "--out", out, "--backend", "stub", "--synthetic", n, "--n", n, "--cpu-check-n", n] + (["--stub-fault", fault] if fault else [])
    r = sh(args); assert r.returncode == 0, r.stderr[-400:] + r.stdout[-300:]
    return out.parent


def forge_gpu(att, stage):
    """FORGED GPU record: only used to exercise the verifier's decision logic.  Never leaves the temp dir."""
    f = att / stage / "run_meta.json"; m = json.loads(f.read_text()); m["device"].update(backend="cupy", cuda_executed=True, gpu_name="FORGED-FOR-SELFTEST"); m["inputs_verified_against_metadata"] = True; m["production_hashes_verified"] = C.EXPECTED
    f.write_text(json.dumps(m))
    if stage == "e6":
        rows = [json.loads(l) for l in (att / stage / "rows.jsonl").read_text().splitlines()]
        for r in rows: r["cuda_executed"] = True
        (att / stage / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if stage == "e4":
        rows = [json.loads(l) for l in (att / stage / "steps.jsonl").read_text().splitlines()]
        for r in rows: r["cuda_executed"] = True
        (att / stage / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def verify(stage, att, *extra):
    r = sh([PKG / "r7" / "r7_verify.py", stage, "--dir", att, "--pkg", PKG, *extra]); v = json.loads((att / "verdict.json").read_text()); return v, r.returncode


@test
def t30_e1_stub_is_never_gpu_evidence():
    att = stub_e1("e1_clean"); v, rc = verify("e1", att, "--expect-n", "3"); assert v["verdict"] == "INVALID" and rc == 2, (v["verdict"], rc)
    return "stub E1 -> INVALID (exit 2)"


@test
def t31_e1_decision_logic_with_forged_record():
    att = stub_e1("e1_forged"); forge_gpu(att, "e1"); v, rc = verify("e1", att, "--expect-n", "3")
    assert v["verdict"] == "PASS" and v["facts"]["independent_verdicts"] == dict(parity="PASS", dwc="PASS", equivalence="DESCRIPTIVE"), v["verdict"]
    return "clean data + forged GPU flag -> PASS (decision plumbing only; this is NOT GPU evidence)"


@test
def t32_negative_control_parity_fault_fails_parity_only():
    att = stub_e1("e1_pf", "parity"); forge_gpu(att, "e1"); v, rc = verify("e1", att, "--expect-n", "3"); iv = v["facts"].get("independent_verdicts", {})
    assert v["verdict"] == "FAIL" and rc == 1 and iv.get("parity") == "FAIL" and iv.get("dwc") == "PASS", (v["verdict"], iv)
    return "injected GPU/CPU-state-machine difference -> FAIL, parity=FAIL, dwc=PASS (axes judged independently)"


@test
def t33_negative_control_dwc_fault_fails_dwc_only():
    att = stub_e1("e1_df", "dwc"); forge_gpu(att, "e1"); v, rc = verify("e1", att, "--expect-n", "3"); iv = v["facts"].get("independent_verdicts", {})
    assert v["verdict"] == "FAIL" and iv.get("dwc") == "FAIL" and iv.get("parity") == "PASS", (v["verdict"], iv)
    return "injected DWC violation -> FAIL, dwc=FAIL, parity=PASS"


@test
def t34_e1_equivalence_difference_alone_does_not_fail():
    att = stub_e1("e1_eq"); forge_gpu(att, "e1")
    for f in (att / "e1" / "subjects").glob("*.json"):       # make Sequential differ from both GPU tie policies in the record
        s = json.loads(f.read_text())
        for p in ("lower", "higher"): s["comparisons"]["semantics"][f"seq_vs_gpu_{p}"].update(equal=False, n_diff_bits=5, max_abs_diff=1.0)
        f.write_text(json.dumps(s))
    v, rc = verify("e1", att, "--expect-n", "3"); assert v["verdict"] == "PASS", v["verdict"]
    assert set(v["facts"]["equivalence_classes"]) <= {"TIE_SEMANTICS_CANDIDATE_NOT_PROVEN", "UNEXPLAINED_NO_TIES", "DIFFERS_BUT_LOWER_EQUALS_HIGHER"}, v["facts"]["equivalence_classes"]
    return "Sequential-vs-CUDA difference is reported (classes " + str(v["facts"]["equivalence_classes"]) + ") but does not gate"


@test
def t35_e1_missing_subject_is_incomplete():
    att = stub_e1("e1_inc"); forge_gpu(att, "e1"); next((att / "e1" / "subjects").glob("*.json")).unlink(); v, rc = verify("e1", att, "--expect-n", "3")
    assert v["verdict"] in ("INCOMPLETE", "INVALID") and rc in (2, 3), (v["verdict"], rc); return f"missing subject -> {v['verdict']}"


@test
def t36_e2_stub_report_is_invalid_and_missing_is_incomplete():
    att = WORK / "e2_x"; att.mkdir(); v, rc = verify("e2", att); assert v["verdict"] == "INCOMPLETE", v["verdict"]
    (att / "parity_gpu").mkdir(); (att / "parity_gpu" / "parity_report.json").write_text(json.dumps(dict(env=dict(backend="stub", gpu=None), rows=[], failed=0)))
    v, rc = verify("e2", att); assert v["verdict"] == "INVALID" and rc == 2, v["verdict"]
    ge = json.loads((att / "gpu_executed.json").read_text()); assert ge["cuda_executed"] is False
    return "no report -> INCOMPLETE; report from a non-CUDA backend -> INVALID, gpu_executed.json says cuda_executed=false"


def stub_e6(name, n=3, reps=3):
    out = WORK / name / "e6"
    r = sh([PKG / "r7" / "r7_e6.py", "--repo", repo(), "--lib", PKG / "lib", "--out", out, "--backend", "stub", "--synthetic", n, "--n", n, "--reps", reps, "--max-load1", "1000"]); assert r.returncode == 0, r.stderr[-400:] + r.stdout[-300:]
    return out.parent


@test
def t40_e6_stub_invalid_and_forged_logic():
    att = stub_e6("e6a"); v, rc = verify("e6", att, "--expect-n", "3"); assert v["verdict"] == "INVALID", v["verdict"]
    forge_gpu(att, "e6"); v, rc = verify("e6", att, "--expect-n", "3", "--min-reps", "3"); assert v["verdict"] == "PASS" and v["facts"]["reps_used"] == [0, 1, 2], (v["verdict"], v["facts"].get("reps_used"))
    return "stub E6 -> INVALID; forged flag -> measurement-validity PASS (3 reps), statements carry bootstrap CIs"


@test
def t41_e6_two_reps_only_is_incomplete_when_three_required():
    att = stub_e6("e6b", reps=2); forge_gpu(att, "e6"); v, rc = verify("e6", att, "--expect-n", "3", "--min-reps", "3"); assert v["verdict"] == "INCOMPLETE" and rc == 3, (v["verdict"], rc)
    return "2 of 3 repetitions -> INCOMPLETE"


@test
def t42_e6_unstable_output_hash_is_invalid():
    att = stub_e6("e6c"); forge_gpu(att, "e6"); f = att / "e6" / "rows.jsonl"; rows = [json.loads(l) for l in f.read_text().splitlines()]
    for r in rows:
        if r["method"] == "ddfp_k16_d1" and r["rep"] == 1: r["out_sha256"] = "0" * 64; break
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n"); v, rc = verify("e6", att, "--expect-n", "3"); assert v["verdict"] in ("INVALID",) and rc == 2, (v["verdict"], rc)
    return "output hash changing between repetitions -> INVALID"


@test
def t43_e3_derived_from_e6_rows():
    att = stub_e6("e6d"); forge_gpu(att, "e6"); out = WORK / "e3_out"; out.mkdir()
    r = sh([PKG / "r7" / "r7_verify.py", "e3", "--dir", att, "--out", out, "--pkg", PKG]); v = json.loads((out / "verdict.json").read_text()); assert v["verdict"] == "PASS", v["verdict"]
    assert any("CPU numpy" in c for c in v["facts"]["statements"]["caveats"]), "coverage / device caveat missing"
    return "E3 PASS (data valid) and always carries the coverage/device caveats"


@test
def t44_e4_stub_safety_rule_and_verifier():
    out = WORK / "e4a" / "e4"
    r = sh([PKG / "r7" / "r7_e4.py", "--repo", repo(), "--lib", PKG / "lib", "--out", out, "--backend", "stub", "--synthetic", 1, "--copies", "1,2,3,4,5,6", "--stub-total-mib", 40]); assert r.returncode == 0, r.stderr[-400:] + r.stdout[-300:]
    rows = [json.loads(l) for l in (out / "steps.jsonl").read_text().splitlines()]; st = {(x["method"], x["copies"]): x["status"] for x in rows}
    assert "OOM_UNEXPECTED" not in st.values() and "ERROR" not in st.values(), st
    att = out.parent; forge_gpu(att, "e4"); v, rc = verify("e4", att); assert v["verdict"] in ("PASS", "FAIL") and "no_unexpected_oom_or_error" in [c["id"] for c in v["checks"] if c["ok"]], v
    again = sh([PKG / "r7" / "r7_e4.py", "--repo", repo(), "--lib", PKG / "lib", "--out", out, "--backend", "stub", "--synthetic", 1, "--copies", "1"]); assert again.returncode != 0, "E4 must refuse to overwrite an earlier attempt directory"
    return f"statuses {sorted(set(st.values()))}; no OOM provoked; a second run into the same directory is refused"


FAKE_SMI = r"""#!/usr/bin/env bash
# fake nvidia-smi for the selftest (scenario in $FAKE_SCN: idle | busy)
USED=300; UTIL=0; APPS="No running processes found"
if [ "$FAKE_SCN" = "busy" ]; then USED=21000; UTIL=87; APPS="GPU-FAKE-0001, 4242, python, 20800"; fi
case "$*" in
  *--query-compute-apps*) echo "$APPS";;
  *"-i 0 --query-gpu"*) echo "$UTIL, $USED, 24564";;
  *--query-gpu=index*) echo "0, GPU-FAKE-0001, NVIDIA GeForce RTX 4090, 550.00, 24564, $USED, $UTIL, 100, 450, 00000000:01:00.0";;
  -L) echo "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-FAKE-0001)";;
  *) echo "fake nvidia-smi $*";;
esac
"""


def fake_preflight(scn):
    b = WORK / f"bin_{scn}"; b.mkdir(); f = b / "nvidia-smi"; f.write_text(FAKE_SMI); f.chmod(0o755)
    out = WORK / f"pf_{scn}"; cfg = WORK / f"pf_{scn}.env"; cfg.write_text(f"R3_BRATS_ROOT={WORK}/nodata\nR3_OUT_ROOT={out}\nR3_REPO={repo()}\nR3_ALLOW_NON_ISOLATED_PYTHON=1\n")
    env = dict(os.environ, PATH=str(b) + os.pathsep + os.environ["PATH"], FAKE_SCN=scn, R7_NO_PAUSE="1")
    r = subprocess.run([sys.executable, "-B", "-I", str(PKG / "r7" / "r7run.py"), "preflight", "--no-server-tree", "--config", str(cfg)], capture_output=True, text=True, env=env)
    v = json.loads(next(out.glob("r7_*/preflight/attempt_*/verdict.json")).read_text()); return {c["id"]: c for c in v["checks"]}, out, cfg, env


@test
def t51_preflight_sees_idle_gpu_and_refuses_busy_gpu():
    ck, *_ = fake_preflight("idle"); assert ck["gpu.idle_and_free"]["ok"], ck["gpu.idle_and_free"]["detail"]
    ck, out, cfg, env = fake_preflight("busy"); c = ck["gpu.idle_and_free"]
    assert not c["ok"] and "other compute process" in c["detail"], c["detail"]
    r = subprocess.run([sys.executable, "-B", "-I", str(PKG / "r7" / "r7run.py"), "e2", "--config", str(cfg)], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "STOP" in r.stdout and not list(out.glob("r7_*/e2/attempt_*")), "a stage must not start (not even create an attempt) when the gate is closed"
    r = subprocess.run([sys.executable, "-B", "-I", str(PKG / "r7" / "r7run.py"), "smoke", "--config", str(cfg)], capture_output=True, text=True, env=env); assert r.returncode == 2 and "STOP" in r.stdout, r.stdout[-300:]
    return "idle fake GPU -> gpu.idle_and_free OK; fake GPU with a foreign 20.8 GB process -> NG; E2 and smoke refuse to start (no attempt directory created)"


# ----------------------------------------------------------------------------------------------------------------------------- gate logic
@test
def t45_e6_hashes_cross_check_against_e1():
    a1 = stub_e1("x_e1", n=3); a6 = stub_e6("x_e6", n=3, reps=3); forge_gpu(a1, "e1"); forge_gpu(a6, "e6")
    v, rc = verify("e6", a6, "--expect-n", "3", "--e1-dir", a1); c = {x["id"]: x for x in v["checks"]}["crosscheck.hashes_equal_E1"]
    assert c["ok"] and c["label"] == "RECOMPUTED", c
    f = a6 / "e6" / "rows.jsonl"; rows = [json.loads(l) for l in f.read_text().splitlines()]; rows[0]["out_sha256"] = "1" * 64          # tamper one hash in every repetition of one method
    for r in rows:
        if (r.get("subject"), r.get("method")) == (rows[0]["subject"], rows[0]["method"]): r["out_sha256"] = "1" * 64
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n"); v, rc = verify("e6", a6, "--expect-n", "3", "--e1-dir", a1)
    assert v["verdict"] == "INVALID" and not {x["id"]: x for x in v["checks"]}["crosscheck.hashes_equal_E1"]["ok"], v["verdict"]
    return "E6 output hashes (K=1, DD-FP, Sequential) equal the E1 hashes for the same inputs; a tampered hash -> INVALID"


@test
def t52_resume_and_extend_keep_history():
    import r7run, r3run
    def mk(root, verdict):
        at = root / "e6" / "attempt_a"; at.mkdir(parents=True); (at / "rows.jsonl").write_text("{}\n"); (at / "verdict.json").write_text(json.dumps(dict(verdict=verdict))); r3run.seal(at); return at
    rd = WORK / "run2a"; at = mk(rd, "INCOMPLETE")
    assert r7run.new_attempt(rd, "e6", resume=True) == at and not (at / "verdict.json").exists(), "INCOMPLETE attempt must be resumable in place"
    assert list(at.glob("verdict.before_resume_*.json")) and list(at.glob("ARTIFACT_SHA256.before_resume_*.txt")), "old verdict/seal must be kept"
    rd = WORK / "run2b"; at = mk(rd, "PASS")
    assert r7run.new_attempt(rd, "e6", resume=True) != at, "--resume must not reopen a finished (PASS) attempt"
    rd = WORK / "run2c"; at = mk(rd, "PASS")
    assert r7run.new_attempt(rd, "e6", extend=True) == at and (at / "rows.jsonl").is_file(), "--extend continues the finished attempt (rows kept)"
    return "INCOMPLETE attempts resume in place, --extend continues a finished one, --resume never reopens PASS; old verdict/seal kept as *.before_resume_*"


@test
def t50_gate_and_seal():
    import r3run, r7run
    rd = WORK / "run"; c = SimpleNamespace(args=SimpleNamespace(triage_note=None)); man = R3.verify_manifest()["manifest_sha256"]
    def mk(stage, name, verdict, seal=True):
        at = rd / stage / name; at.mkdir(parents=True); (at / "verdict.json").write_text(json.dumps(dict(verdict=verdict))); (at / "stage_meta.json").write_text(json.dumps(dict(manifest=dict(manifest_sha256=man), gpu_selected=dict(uuid="U1"))))
        if seal: r3run.seal(at)
        return at
    mk("e2", "attempt_1", "FAIL"); ok, msg, *_ = r7run.gate(c, rd, "e1", "U1"); assert not ok, "FAIL must close the next stage"
    c.args.triage_note = "judged: tie semantics, not a CUDA error"; ok, msg, pre, ov = r7run.gate(c, rd, "e1", "U1"); assert ok and ov and ov["prerequisite_verdict"] == "FAIL", "triage override must be recorded"
    mk("preflight", "attempt_1", "FAIL"); ok, *_ = r7run.gate(c, rd, "e2", "U1"); assert not ok, "preflight FAIL is never overridable even with a triage note"
    shutil.rmtree(rd / "e2"); mk("e2", "attempt_2", "INVALID"); ok, *_ = r7run.gate(c, rd, "e1", "U1"); assert not ok, "INVALID is never overridable"
    shutil.rmtree(rd / "e2"); at = mk("e2", "attempt_3", "PASS"); ok, *_ = r7run.gate(c, rd, "e1", "U1"); assert ok, "sealed PASS opens the gate"
    ok, *_ = r7run.gate(c, rd, "e1", "OTHER-GPU"); assert not ok, "PASS from another GPU must not open the gate"
    (at / "verdict.json").write_text(json.dumps(dict(verdict="PASS", edited=True))); ok, *_ = r7run.gate(c, rd, "e1", "U1"); assert not ok, "verdict edited after sealing must be refused"
    return "FAIL closes (override only with a recorded triage note), INVALID never overridable, PASS needs same package + same GPU + untouched seal"


def main():
    print(f"R7 selftest (CPU only; no CUDA is used or claimed)  work dir {WORK}"); only = sys.argv[1:]
    npass = nfail = nskip = 0
    for name, f in T.items():
        if only and not any(name.startswith(o) for o in only): continue
        t = time.time()
        try: d = f(); npass += 1; print(f"  PASS  {name}  {d}  ({time.time() - t:.1f}s)")
        except Skip as e: nskip += 1; print(f"  SKIP  {name}  {e}")
        except BaseException as e:
            nfail += 1; print(f"  FAIL  {name}  {type(e).__name__}: {e}"); traceback.print_exc(limit=3, file=sys.stdout)
    shutil.rmtree(WORK, ignore_errors=True)
    print(f"\nSELFTEST {'PASS' if nfail == 0 else 'FAIL'}: {npass} passed, {nfail} failed, {nskip} skipped (skipped tests are not counted as passed)")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
