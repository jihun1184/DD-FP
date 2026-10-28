#!/usr/bin/env python3
"""R7 result verifier (offline; numpy + stdlib).  Applies the PRE-REGISTERED decision rules to raw result files and writes verdict.json + RESULT_SUMMARY.md.
It never trusts a summary written by the experiment scripts: every number is recomputed from the raw per-subject / per-run records.

  python r7_verify.py e2|e1|e6|e3|e4 --dir <attempt dir> [--out <dir for verdict files; default = --dir>] [--pkg <package root>] [--e1-dir <e1 attempt dir>] [--expect-n 20]
  Raw data layout: <attempt>/parity_gpu (e2) | <attempt>/e1 | <attempt>/e6 | <attempt>/e4 ; e3 reads the e6 attempt given by --dir.
Verdict values:  PASS | FAIL | INVALID (not usable as GPU evidence / integrity problem) | INCOMPLETE
Exit code: 0 PASS | 1 FAIL | 2 INVALID | 3 INCOMPLETE
STOP rules (a stage gate opens only for PASS):
  E2  : backend must be real cupy with a GPU name, 100/100 runs, every comparison category identical, vendor/fixture/script hashes as in the package.
  E1  : parity (GPU == same-tie CPU state machine, both tie policies) AND dwc (all outputs, GPU checker == independent checker) AND integrity. Equivalence to Sequential FP is
        reported but never gates and is never labelled a CUDA error.
  E6  : measurement validity (complete, deterministic, no foreign GPU process, load ok, hashes equal to E1). Speed statements are reported as measured facts with bootstrap CIs.
  E3  : data validity only; cost statements are pre-registered and always carry the coverage / device caveats.
  E4  : no unexpected OOM/ERROR and at least one completed step per method.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7_common as C

LABEL_RECALC, LABEL_NOTRUN = "RECOMPUTED", "NOT RUN"


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).is_file() else []


def jr(p, d=None):
    try: return json.loads(Path(p).read_text())
    except Exception: return d


class V:
    def __init__(self, stage): self.stage, self.checks, self.facts, self.status = stage, [], {}, None

    def chk(self, cid, ok, detail, required=True, label=LABEL_RECALC):
        self.checks.append(dict(id=cid, ok=bool(ok), required=required, detail=str(detail), label=label)); return bool(ok)

    def req_ok(self): return all(c["ok"] for c in self.checks if c["required"])


def boot(x, B=10000, seed=20261014):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return dict(n=len(x), mean=float(x.mean()), ci95=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], min=float(x.min()), max=float(x.max()))


def finish(v: V, out: Path, status: str, headline: str, extra_md=""):
    v.status = status
    rec = dict(stage=v.stage, verdict=status, headline=headline, checks=v.checks, facts=v.facts, evaluated=time.strftime("%Y-%m-%dT%H:%M:%S%z"), verifier_sha256=C.sha_f(__file__), r7_version=C.R7_VERSION)
    C.jdump(out / "verdict.json", rec)
    md = [f"# {v.stage.upper()} 판정: **{status}**", "", headline, ""] + [f"- [{'OK' if c['ok'] else ('NG' if c['required'] else '--')}] {c['id']}: {c['detail']} ({c['label']})" for c in v.checks]
    (out / "RESULT_SUMMARY.md").write_text("\n".join(md) + "\n" + extra_md, encoding="utf-8")
    print("\n".join(md)); print(f"\n{v.stage.upper()} VERDICT: {status}")
    return {"PASS": 0, "FAIL": 1, "INVALID": 2, "INCOMPLETE": 3}[status]


# --------------------------------------------------------------------------------------------------------------------------------------- E2
def e2(d: Path, pkg: Path, A):
    O = A.out or d; dd = d / "e2"
    v = V("e2"); rp = d / "parity_gpu" / "parity_report.json"; rep = jr(rp); sel = (jr(d / "stage_meta.json", {}) or {}).get("gpu_selected") or {}
    if not v.chk("report.exists", rep is not None, str(rp)): return finish(v, O, "INCOMPLETE", "GPU parity report missing (the stage did not finish).")
    env = rep.get("env", {}); gpu_ok = env.get("backend") == "cupy" and bool(env.get("gpu"))
    C.jdump(O / "gpu_executed.json", dict(cuda_executed=bool(gpu_ok), backend=env.get("backend"), gpu=env.get("gpu"), cupy=env.get("cupy"), runs=rep.get("runs"), selected_gpu=sel, evaluated=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    v.chk("gpu.executed", gpu_ok, f"backend={env.get('backend')} gpu={env.get('gpu')} cupy={env.get('cupy')}  (stub / CPU emulation is never accepted)")
    if sel.get("name"): v.chk("gpu.name_equals_selected", env.get("gpu") == sel["name"], f"report {env.get('gpu')!r} vs runner-selected {sel['name']!r}")
    rows = rep.get("rows", []); fx = jr(pkg / "e2" / "gpu_parity" / "fixtures.json", {})
    exp_ids = [f["id"] for f in fx.get("fixtures", [])]
    v.chk("fixtures.hash", rep.get("fixtures_sha256") == fx.get("sha256_of_list"), f"report {str(rep.get('fixtures_sha256'))[:12]}.. vs package {str(fx.get('sha256_of_list'))[:12]}..")
    v.chk("vendor.hash", rep.get("vendor_sha256_file") == C.sha_f(pkg / "e2" / "gpu_parity" / "vendor" / "VENDOR_SHA256.txt"), "vendored production copies unchanged")
    v.chk("script.hash", rep.get("script_sha256") == C.sha_f(pkg / "e2" / "gpu_parity" / "run_gpu_parity.py"), "run_gpu_parity.py unchanged")
    got = {(r["id"], r["tie"]) for r in rows}; want = {(i, t) for i in exp_ids for t in ("lower", "higher")}
    v.chk("coverage.50x2", len(exp_ids) == 50 and got == want and len(rows) == 100, f"{len(exp_ids)} fixtures x 2 ties, rows={len(rows)}, missing={sorted(want - got)[:3]}")
    bad = [r for r in rows if not r.get("pass")]
    bycat = {}
    for r in bad:
        for k in r.get("bad", {}): bycat[k] = bycat.get(k, 0) + 1
    C.jdump(O / "mismatches.json", dict(n_failed=len(bad), by_category=bycat, rows=[dict(id=r["id"], tie=r["tie"], bad=r.get("bad"), exc=r.get("exc")) for r in bad]))
    v.chk("parity.all_identical", not bad and rep.get("failed") == 0, f"{len(rows) - len(bad)}/{len(rows)} runs identical (failed categories: {bycat or 'none'})")
    v.chk("cpu.deterministic", not rep.get("cpu_nondeterministic"), f"cpu_nondeterministic={rep.get('cpu_nondeterministic')}")
    ties = sum(int(r.get("tie_events_cpu") or 0) for r in rows); cx = [r for r in rows if "3x3x5" in r["id"] or "counter" in r["id"].lower()]
    v.facts = dict(runs=len(rows), failed=len(bad), tie_events_total_cpu_side=ties, rows_with_tie_events=sum(1 for r in rows if (r.get("tie_events_cpu") or 0) > 0), counterexample_rows=[(r["id"], r["tie"], r["pass"]) for r in cx],
                   by_category=bycat, cpu_digest_note=rep.get("cpu_digest_note"), gpu=env.get("gpu"))
    v.chk("coverage.tie_cases_present", v.facts["rows_with_tie_events"] > 0, f"{v.facts['rows_with_tie_events']} runs contain equidistant-level tie events (informational)", required=False)
    if not gpu_ok: return finish(v, O, "INVALID", "Not GPU evidence: the run did not execute on CUDA.")
    if not v.req_ok():
        integ = [c for c in v.checks if c["required"] and not c["ok"] and c["id"] != "parity.all_identical"]
        return finish(v, O, "INVALID" if integ else "FAIL", "E2 failed: " + ("integrity problem" if integ else f"{len(bad)} GPU/CPU mismatching run(s) -> CUDA correctness issue to be resolved first"))
    return finish(v, O, "PASS", "E2: 100/100 GPU runs bit-identical to the CPU state machine in all compared categories (finite evidence on 50 fixtures x 2 tie policies, not a general equivalence).")


# --------------------------------------------------------------------------------------------------------------------------------------- E1
def classify(s):
    c = s["comparisons"]["semantics"]; tl, th = (s["gpu_meta"][p]["equidistant_next_level_tie_events"] or 0 for p in ("lower", "higher"))
    if c["seq_vs_gpu_lower"]["equal"]: return "SEQ_EQUALS_LOWER"
    if c["seq_vs_gpu_higher"]["equal"]: return "SEQ_EQUALS_HIGHER_ONLY"
    if tl == 0 and th == 0: return "UNEXPLAINED_NO_TIES"
    if c["gpu_lower_vs_gpu_higher"]["equal"]: return "DIFFERS_BUT_LOWER_EQUALS_HIGHER"      # the two pure policies give the same output here: neither explains the difference; investigate (tie mixture in insertion order? other cause?)
    return "TIE_SEMANTICS_CANDIDATE_NOT_PROVEN"


def e1(d: Path, pkg: Path, A):
    O = A.out or d; dd = d / "e1"
    v = V("e1"); meta = jr(dd / "run_meta.json", {}); subs = [jr(p) for p in sorted((dd / "subjects").glob("*.json")) if not p.name.endswith(".error.json")]
    errs = list((dd / "subjects").glob("*.error.json")); dev = meta.get("device", {})
    v.chk("gpu.executed", dev.get("backend") == "cupy" and dev.get("cuda_executed") is True and bool(dev.get("gpu_name")), f"backend={dev.get('backend')} gpu={dev.get('gpu_name')}")
    v.chk("subjects.complete", len(subs) == A.expect_n and not errs, f"{len(subs)}/{A.expect_n} subjects, {len(errs)} error record(s)")
    if not subs: return finish(v, O, "INCOMPLETE", "no subject records")
    v.chk("inputs.verified", meta.get("inputs_verified_against_metadata") is True, "input files hash-verified against the recorded cohort metadata")
    v.chk("production.hashes", meta.get("production_hashes_verified") == C.EXPECTED, "6 production hashes verified by the script at start")
    v.chk("no_foreign_gpu_process", not any(s.get("foreign_gpu_procs_after") for s in subs), "no foreign GPU process observed after any subject")
    v.chk("ispan.gpu_equals_cpu", all(s["comparisons"][k]["equal"] for s in subs for k in ("ispan_lo_gpu_vs_cpu", "ispan_hi_gpu_vs_cpu")), "GPU interval-span lattice bit-identical to CPU lattice (precondition of the parity comparison)")
    cl = [s["comparisons"]["semantics"].get("clone_insertion_vs_seq") for s in subs if "clone_insertion_vs_seq" in s["comparisons"]["semantics"]]
    v.chk("cpu_sm.clone_equals_sequential", bool(cl) and all(c["equal"] for c in cl), f"R7 CPU state machine in insertion mode == frozen Sequential FP on {len(cl)} real volume(s)")
    v.chk("prod_k1_equals_direct_lower", all(s["comparisons"]["prod_k1_vs_direct_lower"]["equal"] for s in subs), "production front_propagation_gpu output == seed_aware_boutry_bucket_gpu(tie=lower)")
    # (1) parity
    par_bad = [(s["subject"], p) for s in subs for p in ("lower", "higher") if not s["comparisons"][f"parity_{p}"]["parity_ok"]]
    parity = not par_bad
    v.chk("PARITY.gpu_equals_same_tie_cpu", parity, f"{2 * len(subs) - len(par_bad)}/{2 * len(subs)} (subject x tie policy) bit-identical incl. tie-event counts and level transitions; failing={par_bad[:4]}")
    # (3) dwc
    dwc_bad = [(s["subject"], k) for s in subs for k, x in s["dwc"].items() if x["violations"] != 0 or x.get("gpu_cpu_agree") is False]
    dd_bad = [s["subject"] for s in subs if s.get("ddfp") and s["ddfp"]["dwc"]["violations"] != 0]
    dwc = not dwc_bad and not dd_bad
    v.chk("DWC.global_checks_zero_and_checkers_agree", dwc, f"outputs checked: K1/GPU-higher/Sequential per subject (+DD-FP); violations or checker disagreement: {dwc_bad[:4]} DD-FP:{dd_bad[:4]}")
    # (2) equivalence (never gating)
    cls = {}
    for s in subs: cls.setdefault(classify(s), []).append(s["subject"])
    ndiff = {p: [s["comparisons"]["semantics"][f"seq_vs_gpu_{p}"]["n_diff_bits"] for s in subs] for p in ("lower", "higher")}
    mx = {p: [s["comparisons"]["semantics"][f"seq_vs_gpu_{p}"].get("max_abs_diff") for s in subs] for p in ("lower", "higher")}
    ties = {p: [s["gpu_meta"][p]["equidistant_next_level_tie_events"] for s in subs] for p in ("lower", "higher")}
    dd = [s["ddfp"]["vs_seq"]["n_diff_bits"] for s in subs if s.get("ddfp")]
    v.facts = dict(parity_pass=parity, dwc_pass=dwc, equivalence_classes={k: len(x) for k, x in cls.items()}, equivalence_subjects=cls, seq_vs_gpu_ndiff=ndiff,
                   seq_vs_gpu_maxabs={p: [m for m in x if m] for p, x in mx.items()}, gpu_tie_events=ties, ddfp_vs_seq_ndiff=dd, ddfp_equals_seq_subjects=sum(1 for x in dd if x == 0),
                   hashes={s["subject"]: {k: s["hashes"].get(k) for k in ("gpu_k1_out", "seq_out", "gpu_higher_out")} | {"ddfp_out": (s.get("ddfp") or {}).get("out_sha256")} for s in subs},
                   gpu=dev.get("gpu_name"), note="Equivalence classes are descriptive. TIE_SEMANTICS_CANDIDATE_NOT_PROVEN is a hypothesis about the cause, not a finding; UNEXPLAINED_NO_TIES and DIFFERS_BUT_LOWER_EQUALS_HIGHER need investigation (the latter: pure lower/higher give identical output, so a difference to Sequential FP is not explained by choosing one of them).")
    v.chk("EQUIV.reported_not_gating", True, "Sequential FP vs CUDA-native: " + ", ".join(f"{k}={len(x)}" for k, x in cls.items()) + f"; DD-FP == Sequential in {v.facts['ddfp_equals_seq_subjects']}/{len(dd)}", required=False)
    consist = all(s["comparisons"]["semantics"]["gpu_lower_vs_gpu_higher"]["equal"] for s in subs if (s["gpu_meta"]["lower"]["equidistant_next_level_tie_events"] or 0) == 0 and (s["gpu_meta"]["higher"]["equidistant_next_level_tie_events"] or 0) == 0)
    v.chk("consistency.no_ties_implies_lower_equals_higher", consist, "subjects without tie events: GPU lower output == GPU higher output", required=False)
    if dev.get("backend") != "cupy" or not v.checks[0]["ok"]: return finish(v, O, "INVALID", "Not GPU evidence.")
    integ = [c for c in v.checks if c["required"] and not c["ok"] and not c["id"].startswith(("PARITY", "DWC"))]
    if integ: return finish(v, O, "INCOMPLETE" if not v.checks[1]["ok"] else "INVALID", "integrity/completeness problem: " + ", ".join(c["id"] for c in integ))
    v.facts["independent_verdicts"] = dict(parity="PASS" if parity else "FAIL", dwc="PASS" if dwc else "FAIL", equivalence="DESCRIPTIVE")
    if parity and dwc: return finish(v, O, "PASS", f"E1 gate passed: parity PASS, DWC PASS (independent). Sequential-FP equivalence classes: {v.facts['equivalence_classes']}.")
    return finish(v, O, "FAIL", f"E1 STOP: parity={'PASS' if parity else 'FAIL'}, dwc={'PASS' if dwc else 'FAIL'} (judged independently). Resolve the cause before E6.")


# --------------------------------------------------------------------------------------------------------------------------------------- E6
def table(rows, reps):
    """subject -> method -> {key: mean over reps}"""
    t = {}
    for r in rows:
        if r.get("rep", -1) in reps and r.get("method") in C.METHODS:
            t.setdefault(r["subject"], {}).setdefault(r["method"], []).append(r)
    return t


def e6(d: Path, pkg: Path, A):
    O = A.out or d; dd = d / "e6"
    v = V("e6"); meta = jr(dd / "run_meta.json", {}); rows = [r for r in jl(dd / "rows.jsonl") if r.get("method") in C.METHODS and r.get("rep", -1) >= 0]; mem = [r for r in jl(dd / "rows.jsonl") if r.get("method") == "mem"]
    dev = meta.get("device", {}); subs = sorted({r["subject"] for r in rows}); reps_all = sorted({r["rep"] for r in rows})
    have = {s: {m: {r["rep"] for r in rows if r["subject"] == s and r["method"] == m} for m in C.METHODS} for s in subs}
    reps = sorted(set.intersection(*[have[s][m] for s in subs for m in C.METHODS])) if subs else []
    v.chk("gpu.executed", dev.get("backend") == "cupy" and dev.get("cuda_executed") is True and bool(dev.get("gpu_name")) and all(r.get("cuda_executed") for r in rows), f"backend={dev.get('backend')} gpu={dev.get('gpu_name')}")
    v.chk("complete.subjects_x_reps_x_methods", len(subs) == A.expect_n and len(reps) >= A.min_reps, f"{len(subs)} subjects, repetitions common to all methods/subjects: {reps} (need >= {A.min_reps})")
    if not subs or not reps: return finish(v, O, "INCOMPLETE", "no complete repetition")
    t = table(rows, set(reps))
    v.chk("no_foreign_gpu_process", not any(r.get("foreign_gpu_procs_after") for r in rows), "no foreign GPU process observed after any run")
    lim = (meta.get("host", {}).get("cpu_count") or 1) * 0.5
    v.chk("cpu.not_contended", all((r.get("load1_before") or 0) <= max(lim, 1.0) + 1.0 for r in rows), f"1-min load before each run <= {max(lim, 1.0) + 1.0:.1f} (note: the run itself adds ~1)", required=False)
    det = {m: all(len({r["out_sha256"] for r in t[s][m]}) == 1 for s in subs) for m in C.METHODS}
    v.chk("deterministic.output_hash_stable_across_reps", all(det.values()), f"{det}")
    e1d = Path(A.e1_dir) if A.e1_dir else None
    if e1d and (e1d / "e1" / "subjects").is_dir():
        mm = []
        for s in subs:
            h = (jr(e1d / "e1" / "subjects" / f"{s}.json", {}) or {}).get("hashes", {}); ddr = (jr(e1d / "e1" / "subjects" / f"{s}.json", {}) or {}).get("ddfp", {})
            for m, key in (("gpu_k1", h.get("gpu_k1_out")), ("seq_fp_cpu", h.get("seq_out")), ("ddfp_k16_d1", ddr.get("out_sha256"))):
                if key and t[s][m][0]["out_sha256"] != key: mm.append((s, m))
        v.chk("crosscheck.hashes_equal_E1", not mm, f"E6 output hashes equal E1 hashes for K1/Sequential/DD-FP ({mm[:3]})")
    else: v.chk("crosscheck.hashes_equal_E1", False, "E1 attempt directory not supplied", required=False, label=LABEL_NOTRUN)
    def ms(m, key="t_algorithm_s"): return np.array([np.mean([r[key] for r in t[s][m]]) for s in subs])
    k1, dd_, sq = ms("gpu_k1"), ms("ddfp_k16_d1"), ms("seq_fp_cpu"); dd_rec = np.array([np.mean([r["t_recorded_equiv_s"] for r in t[s]["ddfp_k16_d1"]]) for s in subs]) if all("t_recorded_equiv_s" in r for s in subs for r in t[s]["ddfp_k16_d1"]) else None
    cv = {m: [float(np.std([r["t_algorithm_s"] for r in t[s][m]], ddof=1) / np.mean([r["t_algorithm_s"] for r in t[s][m]])) for s in subs] for m in C.METHODS} if len(reps) > 1 else {}
    medcv = {m: float(np.median(x)) for m, x in cv.items()}; maxcv = {m: float(np.max(x)) for m, x in cv.items()}
    v.facts = dict(reps_used=reps, n_subjects=len(subs), host_cpu=meta.get("host", {}).get("cpu_model"), gpu=dev.get("gpu_name"), mean_seconds={m: float(x.mean()) for m, x in (("gpu_k1", k1), ("ddfp_k16_d1", dd_), ("seq_fp_cpu", sq))},
                   sd_across_subjects={m: float(x.std(ddof=1)) for m, x in (("gpu_k1", k1), ("ddfp_k16_d1", dd_), ("seq_fp_cpu", sq))}, median_cv_between_reps=medcv, max_cv_between_reps=maxcv,
                   ratio_seq_over_k1=boot(sq / k1), ratio_ddfp_over_k1=boot(dd_ / k1), ratio_ddfp_recorded_equiv_over_k1=(boot(dd_rec / k1) if dd_rec is not None else None),
                   phases_mean={m: {p: float(np.mean([np.mean([r[p] for r in t[s][m]]) for s in subs])) for p in ("t_prep", "t_fp", "t_d2h", "t_ibi", "t_out") if all(p in r for s in subs for r in t[s][m])} for m in C.METHODS})
    f = v.facts; sq1, dk1 = f["ratio_seq_over_k1"], f["ratio_ddfp_over_k1"]
    f["statements"] = dict(K1_faster_than_Sequential_same_server=("YES" if sq1["ci95"][0] > 1 else "NOT ESTABLISHED"), DDFP_K16_slower_than_K1=("YES" if dk1["ci95"][0] > 1 else ("NO: DD-FP faster" if dk1["ci95"][1] < 1 else "NOT DISTINGUISHABLE")),
                           basis="bootstrap 95% CI (B=10000, seed 20261014) of the subject-level mean time ratio; subject mean over repetitions", scope="one server, one GPU model, 20 subjects, single-thread Python Sequential FP baseline (not an optimised CPU implementation)")
    f["more_repetitions_recommended"] = bool(medcv and (max(medcv.values()) > 0.05 or max(maxcv.values()) > 0.15))
    if mem:
        pk = {m: [x[m]["peak_vram_delta_mib"] for x in mem if m in x] for m in ("gpu_k1", "ddfp_k16_d1")}
        f["memory_pass_peak_vram_delta_mib"] = {m: dict(mean=float(np.mean(x)), max=float(np.max(x)), n=len(x)) for m, x in pk.items() if x}
    f["host_peak_rss_mib_seq"] = float(np.mean([np.mean([r.get("vmhwm", np.nan) for r in t[s]["seq_fp_cpu"]]) for s in subs]))
    if not v.checks[0]["ok"]: return finish(v, O, "INVALID", "Not GPU evidence.")
    if not v.req_ok(): return finish(v, O, "INCOMPLETE" if not v.checks[1]["ok"] else "INVALID", "protocol validity problem: " + ", ".join(c["id"] for c in v.checks if c["required"] and not c["ok"]))
    return finish(v, O, "PASS", f"E6 measurement valid ({len(reps)} reps x {len(subs)} subjects). Seq/K1 = {sq1['mean']:.2f}x (CI {sq1['ci95'][0]:.2f}-{sq1['ci95'][1]:.2f}); DD-FP/K1 = {dk1['mean']:.2f}x (CI {dk1['ci95'][0]:.2f}-{dk1['ci95'][1]:.2f}). "
                  f"K1 faster than Sequential: {f['statements']['K1_faster_than_Sequential_same_server']}; DD-FP slower than K1: {f['statements']['DDFP_K16_slower_than_K1']}; more reps recommended: {f['more_repetitions_recommended']}")


# --------------------------------------------------------------------------------------------------------------------------------------- E3 (derived from E6 rows)
def e3(d: Path, pkg: Path, A):
    O = A.out or d; dd = d / "e6"
    v = V("e3"); rows = [r for r in jl(dd / "rows.jsonl") if r.get("method") in ("gpu_k1", "ddfp_k16_d1") and r.get("rep", -1) >= 0]; dev = jr(dd / "run_meta.json", {}).get("device", {})
    v.chk("gpu.executed", dev.get("backend") == "cupy" and dev.get("cuda_executed") is True, f"backend={dev.get('backend')}")
    need = lambda r: all(k in r for k in (("t_h2d_s", "t_global_s") + (("t_seam_s",) if r["method"] == "ddfp_k16_d1" else ())))
    v.chk("data.verification_costs_present", bool(rows) and all(need(r) for r in rows), f"{sum(need(r) for r in rows)}/{len(rows)} GPU rows carry verification costs")
    subs = sorted({r["subject"] for r in rows})
    if not v.req_ok() or not subs: return finish(v, O, "INCOMPLETE" if dev.get("backend") == "cupy" else "INVALID", "verification-cost data missing or not GPU")
    g = lambda m, k: np.array([np.mean([r[k] for r in rows if r["subject"] == s and r["method"] == m]) for s in subs])
    seam, h2d_d, glob_d, ibi = g("ddfp_k16_d1", "t_seam_s"), g("ddfp_k16_d1", "t_h2d_s"), g("ddfp_k16_d1", "t_global_s"), g("ddfp_k16_d1", "t_algorithm_s")
    h2d_k, glob_k, k1 = g("gpu_k1", "t_h2d_s"), g("gpu_k1", "t_global_s"), g("gpu_k1", "t_algorithm_s")
    viol = sum(int(r.get("global_violations", 0)) for r in rows); sviol = sum(int(r.get("seam_violations", 0)) for r in rows if r["method"] == "ddfp_k16_d1")
    v.chk("data.global_violations_zero", viol == 0, f"global face+cube violations over all verified outputs: {viol}; seam violations: {sviol}", required=False)
    f = v.facts = dict(n_subjects=len(subs), mean_seconds=dict(seam_scan_cpu_numpy=float(seam.mean()), global_check_ddfp_output_gpu=float((h2d_d + glob_d).mean()), global_kernel_only_ddfp=float(glob_d.mean()), h2d_ddfp=float(h2d_d.mean()),
                       global_check_k1_output_gpu=float((h2d_k + glob_k).mean()), ddfp_ibi=float(ibi.mean()), k1_algorithm=float(k1.mean())),
                       ratio_global_over_seam=boot((h2d_d + glob_d) / seam), ratio_globalkernel_over_seam=boot(glob_d / seam),
                       cert_inclusive=dict(ddfp_ibi_plus_seam_over_k1_plus_global=boot((ibi + seam) / (k1 + h2d_k + glob_k)), ddfp_ibi_plus_global_over_k1_plus_global=boot((ibi + h2d_d + glob_d) / (k1 + h2d_k + glob_k))))
    ci = f["cert_inclusive"]["ddfp_ibi_plus_seam_over_k1_plus_global"]["ci95"]; rg = f["ratio_global_over_seam"]["ci95"]
    f["statements"] = dict(seam_scan_cheaper_than_global_check=("YES (partial coverage, different device)" if rg[0] > 1 else "NOT ESTABLISHED"),
                           ddfp_certification_inclusive_advantage_over_K1_plus_global=("CANDIDATE (against global check; seam covers only xz/yz faces at cut planes)" if ci[1] < 1 else "NO"),
                           caveats=["seam scan = production count_boundary_violations: CPU numpy, xz/yz face blocks at the cut planes only; global check = production verify_dwc_gpu: GPU, all face + cube blocks, all orientations",
                                    "the two checks do not certify the same property; full block-wise certification (all tile-interior + cross-owner blocks) is not implemented in production",
                                    "DD-FP output is not equal to Sequential FP/K=1 output (see E1)"])
    return finish(v, O, "PASS", f"E3 data valid. mean seam scan {seam.mean():.3f}s vs global check {(h2d_d + glob_d).mean():.3f}s on DD-FP output; statements: {f['statements']['seam_scan_cheaper_than_global_check']}; certification-inclusive: {f['statements']['ddfp_certification_inclusive_advantage_over_K1_plus_global']}")


# --------------------------------------------------------------------------------------------------------------------------------------- E4
def e4(d: Path, pkg: Path, A):
    O = A.out or d; dd = d / "e4"
    v = V("e4"); rows = jl(dd / "steps.jsonl"); dev = jr(dd / "run_meta.json", {}).get("device", {})
    v.chk("gpu.executed", dev.get("backend") == "cupy" and dev.get("cuda_executed") is True, f"backend={dev.get('backend')}")
    bym = {}
    for r in rows: bym.setdefault(r["method"], []).append(r)
    comp = {m: [r for r in x if r["status"] == "COMPLETED"] for m, x in bym.items()}
    v.chk("no_unexpected_oom_or_error", all(r["status"] in ("COMPLETED", "SKIPPED_BY_SAFETY_RULE") for r in rows), f"statuses: { {m: [r['status'] for r in x] for m, x in bym.items()} }")
    v.chk("each_method_completed_a_step", len(comp) >= 2 and all(comp.values()), "K1 and DD-FP completed at least the first step")
    v.chk("outputs_sane", all(r.get("out_finite") and r.get("out_shape") == r.get("expected_out_shape") for x in comp.values() for r in x), "all completed outputs finite with the expected shape")
    v.chk("no_foreign_gpu_process", not any(r.get("foreign_gpu_procs_after") for r in rows), "no foreign GPU process observed")
    if not rows or not v.checks[0]["ok"]: return finish(v, O, "INVALID" if rows else "INCOMPLETE", "E4 not GPU evidence or empty")
    byk = {m: {r["copies"]: r for r in x} for m, x in comp.items()}; common = sorted(set(byk.get("gpu_k1", {})) & set(byk.get("ddfp_k16_d1", {})))
    ratio = {k: byk["ddfp_k16_d1"][k]["peak_vram_delta_mib"] / byk["gpu_k1"][k]["peak_vram_delta_mib"] for k in common if byk["gpu_k1"][k]["peak_vram_delta_mib"] > 0}
    maxc = {m: max((r["expanded_cells"] for r in x), default=0) for m, x in comp.items()}
    stop = {m: [(r["copies"], r["admission"]) for r in x if r["status"] == "SKIPPED_BY_SAFETY_RULE"] for m, x in bym.items()}
    f = v.facts = dict(common_sizes=common, vram_ratio_ddfp_over_k1=ratio, peak_vram_delta_mib={m: {k: r["peak_vram_delta_mib"] for k, r in x.items()} for m, x in byk.items()}, peak_rss_mib={m: {k: r["peak_rss_mib"] for k, r in x.items()} for m, x in byk.items()},
                       seconds={m: {k: r["seconds"] for k, r in x.items()} for m, x in byk.items()}, max_completed_cells=maxc, stopped_by_safety_rule=stop,
                       statements=dict(memory_advantage_candidate=bool(sum(1 for x in ratio.values() if x <= 0.8) >= 2), larger_max_size_candidate=bool(maxc.get("ddfp_k16_d1", 0) > maxc.get("gpu_k1", 0)),
                                       note="pre-registered thresholds: ratio <= 0.8 on >= 2 common sizes; max completed size is a LOWER bound set by the safety rule, one subject tiled along z"))
    return finish(v, O, "PASS" if v.req_ok() else "FAIL", f"E4: completed sizes {dict((m, sorted(x)) for m, x in byk.items())}; DD-FP/K1 peak-VRAM ratios {({k: round(x, 2) for k, x in ratio.items()})}; statements {f['statements']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["e2", "e1", "e6", "e3", "e4"]); ap.add_argument("--dir", type=Path, required=True); ap.add_argument("--pkg", type=Path, default=C.PKG)
    ap.add_argument("--out", type=Path); ap.add_argument("--e1-dir", type=Path); ap.add_argument("--expect-n", type=int, default=20); ap.add_argument("--min-reps", type=int, default=3)
    A = ap.parse_args(); O = A.out or A.dir; O.mkdir(parents=True, exist_ok=True)
    try: return {"e2": e2, "e1": e1, "e6": e6, "e3": e3, "e4": e4}[A.stage](A.dir, A.pkg, A)
    except Exception as e:   # a verifier crash must never leave the stage without a verdict
        import traceback; v = V(A.stage); v.chk("verifier.crashed", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}")
        return finish(v, O, "INVALID", "result verifier crashed -> treat as INVALID and send the folder")


if __name__ == "__main__":
    sys.exit(main())
