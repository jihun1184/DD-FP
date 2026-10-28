#!/usr/bin/env python3
"""
test_g5_v2.py -- tests of the G5 v2 harness on the CPU shim (no GPU, no evidence about speed).

What is tested:  decision rules (every verdict branch, with fabricated CUDA-flagged rows), status classification, completeness check, execution-order
balance, VRAM poller, bootstrap determinism, generic pile v2 equivalence + typed errors, and the real runner with injected faults
(output corruption, capacity failure, implementation error, CUDA fault, AF regression, frozen failure) and the stage-1/stage-2 gate.
Run:  python3 -I comparison/test_g5_v2.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "comparison"))
import af_worklist as AF  # noqa: E402
import g5_rules as R  # noqa: E402
import generic_worklists_v2 as GW  # noqa: E402
import harness as Hh  # noqa: E402
import run_g5_v2 as RUN  # noqa: E402

RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)


def fab_rows(subjects, reps, t, status=None, overrides=None):
    """fabricated timed rows; t[m] = seconds (callable (subject, rep) -> float or float)."""
    rows = []
    for si, s in enumerate(subjects):
        for rep in range(reps):
            order = RUN.order_for(si, rep, 1)
            for pos, m in enumerate(order):
                tv = t[m](s, rep) if callable(t[m]) else t[m]
                row = dict(subject=s, rep=rep, method=m, order_position=pos, phase="timed", status=R.OK, t_total_s=tv)
                if overrides and (s, rep, m) in overrides:
                    row.update(overrides[(s, rep, m)])
                rows.append(row)
    return rows


def meta_for(subjects, reps, cuda=True, expected=None):
    return dict(stage=2, cuda_executed=cuda, reps=reps, subjects=subjects, expected_subjects=expected if expected is not None else subjects)


def rules_tests():
    S = [f"s{i}" for i in range(20)]
    jit = lambda base: (lambda s, rep: base * (1 + 0.01 * ((hash((s, rep)) % 5) - 2)))  # noqa: E731
    base = dict(frozen=jit(14.0), af=jit(3.0))
    V = lambda rows, **k: R.decide(rows, meta_for(S, 6, **k))  # noqa: E731
    v = V(fab_rows(S, 6, dict(base, pile=jit(2.0))))
    check("rules: PILE-FASTER (r=1.5, tight)", v["verdict"] == "PILE-FASTER", v["verdict"])
    v = V(fab_rows(S, 6, dict(base, pile=jit(4.5))))
    check("rules: AF-FASTER (r=0.67)", v["verdict"] == "AF-FASTER", v["verdict"])
    v = V(fab_rows(S, 6, dict(base, pile=jit(3.0))))
    check("rules: EQUIVALENT (r=1.0)", v["verdict"] == "EQUIVALENT", v["verdict"])
    mixed = dict(base, pile=lambda s, rep: 2.0 if int(s[1:]) % 2 == 0 else 4.5)
    v = V(fab_rows(S, 6, mixed))
    check("rules: MIXED-SUBJECT-DEPENDENT", v["verdict"] == "MIXED-SUBJECT-DEPENDENT", v["verdict"])
    near = dict(base, pile=jit(2.65))  # r = 1.13: median above margin but CI may include it -> must not be an unconditional PILE-FASTER claim
    v = V(fab_rows(S, 6, near))
    check("rules: near-margin ratio gives a definite class", v["verdict"] in ("PILE-FASTER", "INDETERMINATE (CI straddles the equivalence margin)", "EQUIVALENT"), v["verdict"])
    straddle = dict(base, pile=lambda s, rep: 3.0 / (1.10 + (0.05 if int(s[1:]) % 2 else -0.05)))
    v = V(fab_rows(S, 6, straddle))
    check("rules: straddling CI -> INDETERMINATE", v["verdict"].startswith(("INDETERMINATE", "MIXED")), v["verdict"])
    noisy = dict(base, pile=lambda s, rep: 2.0 * (1.5 if rep % 2 else 1.0))
    v = V(fab_rows(S, 6, noisy))
    check("rules: noisy pile timing -> INDETERMINATE-NOISY", v["verdict"].startswith("INDETERMINATE-NOISY"), v["verdict"])
    # failure classes
    ov = {("s3", 2, "pile"): dict(status=R.CORRECTNESS_FAIL, t_total_s=None)}
    v = V(fab_rows(S, 6, dict(base, pile=jit(2.0)), overrides=ov))
    check("rules: PILE correctness failure -> implementation-failure verdict, no AF-NECESSARY",
          v["verdict"].startswith("PILE-IMPLEMENTATION-FAILURE") and "AF-NECESSARY" not in json.dumps(v), v["verdict"])
    ov = {("s3", 2, "pile"): dict(status=R.IMPL_ERROR, t_total_s=None)}
    v = V(fab_rows(S, 6, dict(base, pile=jit(2.0)), overrides=ov))
    check("rules: PILE implementation error -> implementation-failure verdict", v["verdict"].startswith("PILE-IMPLEMENTATION-FAILURE"), v["verdict"])
    ov = {("s3", r, "pile"): dict(status=R.RESOURCE_FAIL, t_total_s=None) for r in range(6)}
    v = V(fab_rows(S, 6, dict(base, pile=jit(2.0)), overrides=ov))
    check("rules: PILE resource failure -> resource-limited verdict (not semantic)", v["verdict"].startswith("PILE-RESOURCE-LIMITED") and "speed_descriptive" in v, v["verdict"])
    ov = {("s3", 2, "af"): dict(status=R.CORRECTNESS_FAIL, t_total_s=None)}
    v = V(fab_rows(S, 6, dict(base, pile=jit(2.0)), overrides=ov))
    check("rules: AF failure -> INVALID-AF-REGRESSION", v["verdict"].startswith("INVALID-AF-REGRESSION"), v["verdict"])
    ov = {("s3", 2, "frozen"): dict(status=R.RESOURCE_FAIL, t_total_s=None)}
    v = V(fab_rows(S, 6, dict(base, pile=jit(2.0)), overrides=ov))
    check("rules: frozen failure -> INVALID-REFERENCE-FAILURE", v["verdict"].startswith("INVALID-REFERENCE"), v["verdict"])
    ov = {("s3", 2, "pile"): dict(status=R.CUDA_FAULT, t_total_s=None)}
    v = V(fab_rows(S, 6, dict(base, pile=jit(2.0)), overrides=ov))
    check("rules: CUDA fault -> ABORTED", v["verdict"].startswith("ABORTED-CUDA-FAULT"), v["verdict"])
    rows = fab_rows(S, 6, dict(base, pile=jit(2.0)))
    v = V([r for r in rows if not (r["subject"] == "s5" and r["rep"] == 4 and r["method"] == "af")])
    check("rules: missing row -> INVALID-RUN", v["verdict"].startswith("INVALID-RUN"), v["verdict"])
    v = V(rows + [dict(rows[0])])
    check("rules: duplicate row -> INVALID-RUN", v["verdict"].startswith("INVALID-RUN"), v["verdict"])
    v = R.decide(fab_rows(S[:10], 6, dict(base, pile=jit(2.0))), meta_for(S[:10], 6, expected=S))
    check("rules: fewer than 20 / different cohort -> NOT-DECIDABLE", v["verdict"].startswith("NOT-DECIDABLE"), v["verdict"])
    v = R.decide(fab_rows(S, 6, dict(base, pile=jit(2.0))), meta_for(S, 6, cuda=False))
    check("rules: shim run is never a verdict", v["verdict"].startswith("NOT-MEASURED"), v["verdict"])
    ov = {("s3", -1, "pile"): dict(status=R.IMPL_ERROR, t_total_s=None, phase="profile")}
    v = V(fab_rows(S, 6, dict(base, pile=jit(2.0))) + [dict(subject="s3", rep=-1, method="pile", order_position=0, phase="profile", status=R.IMPL_ERROR, t_total_s=None)])
    check("rules: a failing intrusive profile pass never changes the verdict", v["verdict"] == "PILE-FASTER", v["verdict"])
    # exceptions
    check("classify: capacity", R.classify_exception(R.GenericResourceError("x")) == R.RESOURCE_FAIL)
    check("classify: OOM by name", R.classify_exception(type("OutOfMemoryError", (Exception,), {})("x")) == R.RESOURCE_FAIL)
    check("classify: invariant", R.classify_exception(R.GenericInvariantError("x")) == R.IMPL_ERROR)
    check("classify: CUDA", R.classify_exception(type("CUDARuntimeError", (Exception,), {})("x")) == R.CUDA_FAULT)
    check("classify: compile error is an implementation error", R.classify_exception(type("CompileException", (Exception,), {})("x")) == R.IMPL_ERROR)
    # statistics / design
    a = R.bootstrap_median_ci(list(np.linspace(0.9, 1.4, 20))); b = R.bootstrap_median_ci(list(np.linspace(0.9, 1.4, 20)))
    check("bootstrap: deterministic", a == b and a[0] < a[1], str(a))
    for reps in (3, 6, 12):
        seen = {}
        for si in range(5):
            orders = [tuple(RUN.order_for(si, r, 7)) for r in range(reps)]
            pos = {(m, p) for o in orders for p, m in enumerate(o)}
            seen[si] = (len(pos) == 9, len(set(orders)) == min(reps, 6))
        check(f"order design reps={reps}: every method at every position; {min(reps,6)} distinct orders", all(a and b for a, b in seen.values()), str(seen))
    fake = iter([(100 * 2**20, 0), (100 * 2**20, 0), (60 * 2**20, 0), (80 * 2**20, 0)] + [(90 * 2**20, 0)] * 10000)
    vp = RUN.VramPoll(lambda: next(fake), interval=0.0005)
    with vp:
        import time as _t; _t.sleep(0.01)
    check("VramPoll: peak delta from the minimum free", abs(vp.result()["peak_vram_delta_mib"] - 40.0) < 1e-9 and vp.result()["samples"] >= 3, str(vp.result()))


def pile_v2_tests(ncases):
    cp = __import__("fakecupy").FakeCupy(seed=11, threads=4)
    F = AF.load_frozen()
    rng = random.Random(77); nprng = np.random.default_rng(77)
    bad = 0; typed = 0; tot = 0
    for i in range(ncases):
        kind, lo, hi, linf = Hh.gen_case(rng, nprng, i)
        ref = F.boutry_bucket_gpu(cp, lo, hi, linf, "lower", 64)
        got = GW.generic_pile_worklist(cp, lo, hi, linf, "nearest-lower", 64, frozen=F)
        tot += 1; bad += int(bool(RUN.compare(ref, got)))
        try:
            GW.generic_pile_worklist(cp, lo, hi, linf, "nearest-lower", 64, list_cap=2, pile_cap=2, frozen=F)
        except R.GenericResourceError:
            typed += 1
        except Exception:  # noqa: BLE001
            typed -= 10**6
        except BaseException:
            raise
    check(f"generic pile v2 == frozen on {tot} synthetic cases (output, aux, counters, transitions, closure)", bad == 0, f"{bad} mismatches")
    check("generic pile v2: tiny caps raise the typed resource error (never a bare RuntimeError)", typed > 0, str(typed))
    g = got[1]["gw"]
    check("generic pile v2 reports allocation and occupancy", g["alloc"]["total_mib"] > 0 and 0 <= g["occupancy"]["max_pile_over_cap"] <= 1 and "frontier" not in g["alloc"]["bytes"])
    # the compare() helper must see differences it is supposed to see
    ref = F.boutry_bucket_gpu(cp, lo, hi, linf, "lower", 64)
    got = GW.generic_pile_worklist(cp, lo, hi, linf, "nearest-lower", 64, frozen=F)
    for what in ("out", "aux", "meta"):
        g2 = (got[0].copy(), dict(got[1]), {k: v.copy() for k, v in got[2].items()})
        if what == "out":
            g2[0].flat[0] = -1.0
        elif what == "aux":
            g2[2]["scheduled_level"].flat[0] += 1
        else:
            g2[1]["level_transitions"] = g2[1]["level_transitions"] + [[0, 0]]
        check(f"compare() detects a changed {what}", len(RUN.compare(ref, g2)) >= 1)
    g2 = (got[0], dict(got[1]), dict(got[2])); g2[1]["closure_rounds_by_level"] = {"x": 1}
    check("compare() detects changed closure rounds", len(RUN.compare(ref, g2)) == 1)


def runner_tests(workdir: Path):
    py = [sys.executable, "-I", str(ROOT / "comparison" / "run_g5_v2.py"), "--backend", "shim", "--synthetic", "2", "--synthetic-size", "8", "--reps", "3"]

    def go(name, stage, extra, expect_rc=None):
        out = workdir / name
        p = subprocess.run(py + ["--stage", str(stage), "--out", str(out)] + extra, capture_output=True, text=True, timeout=900)
        rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()] if (out / "rows.jsonl").exists() else []
        summ = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else None
        return p, rows, summ

    def forced(rows, subjects, reps=3):
        return R.decide(rows, dict(stage=1, cuda_executed=True, reps=reps, subjects=subjects, expected_subjects=None))

    # clean
    p, rows, s = go("clean", 1, [])
    check("runner: clean shim run completes, shim is flagged NOT-MEASURED", p.returncode == 3 and s and s["decision"]["verdict"].startswith("NOT-MEASURED"), p.stdout[-300:] + p.stderr[-300:])
    ph = {r["phase"] for r in rows}
    check("runner: warm-up/check/timed/memory/profile rows are separate phases", {"warmup", "check", "timed", "memory", "profile"} <= ph, str(ph))
    timed = [r for r in rows if r["phase"] == "timed"]
    check("runner: timed rows are exactly reps 0..2 x 3 methods, all with a time", len(timed) == 9 and {r["rep"] for r in timed} == {0, 1, 2} and all(r["t_total_s"] for r in timed))
    check("runner: warm-up/check/memory/profile rows are never reps of the timed design", all(r["rep"] in (-1, 0, 1) for r in rows if r["phase"] != "timed") and all(r["subject"] == "warmup" for r in rows if r["phase"] == "warmup"))
    v = forced([r for r in rows], ["synthetic0"])
    check("runner: clean rows pass the (CUDA-forced) decision logic without failures", v["verdict"].startswith(("NOT-DECIDABLE", "EQUIVALENT", "PILE-FASTER", "AF-FASTER", "INDETERMINATE", "MIXED")), v["verdict"])
    check("runner: every timed row has exactly one entry", R.completeness_problems(rows, ["synthetic0"], 3) == [])
    check("runner: stage-1 gate is not passed on the shim (no CUDA, no VRAM probe)", not s["stage1_gate"]["passed"] and not (workdir / "clean" / "STAGE1_PASS.json").exists())
    # injected faults
    p, rows, s = go("pile_corrupt", 1, ["--inject", "pile:corrupt:1"])
    v = forced(rows, ["synthetic0"])
    check("fault: corrupted PILE output -> CORRECTNESS_FAIL row + implementation-failure verdict, run continues",
          v["verdict"].startswith("PILE-IMPLEMENTATION-FAILURE") and any(r["status"] == R.CORRECTNESS_FAIL and r["method"] == "pile" and r["rep"] == 1 for r in rows)
          and sum(r["phase"] == "timed" for r in rows) == 9, v["verdict"])
    p, rows, s = go("pile_oom", 1, ["--inject", "pile:oom"])
    v = forced(rows, ["synthetic0"])
    check("fault: PILE capacity failure -> RESOURCE_FAIL (warm-up stops the run before timing; not a semantic verdict)",
          any(r["status"] == R.RESOURCE_FAIL for r in rows) and v["verdict"].startswith("PILE-RESOURCE-LIMITED"), v["verdict"])
    p, rows, s = go("pile_impl", 1, ["--inject", "pile:impl"])
    v = forced(rows, ["synthetic0"])
    check("fault: PILE implementation error -> IMPL_ERROR", v["verdict"].startswith("PILE-IMPLEMENTATION-FAILURE") and any(r["status"] == R.IMPL_ERROR for r in rows), v["verdict"])
    p, rows, s = go("af_corrupt", 1, ["--inject", "af:corrupt"])
    v = forced(rows, ["synthetic0"])
    check("fault: AF disagreement aborts the run and is called an AF regression", p.returncode == 4 and v["verdict"].startswith("INVALID-AF-REGRESSION"), f"rc={p.returncode} {v['verdict']}")
    p, rows, s = go("frozen_oom", 1, ["--inject", "frozen:oom"])
    v = forced(rows, ["synthetic0"])
    check("fault: frozen failure aborts (no reference)", p.returncode == 4 and v["verdict"].startswith("INVALID-REFERENCE"), f"rc={p.returncode} {v['verdict']}")
    p, rows, s = go("cuda", 1, ["--inject", "pile:cudafault:0"])
    v = forced(rows, ["synthetic0"])
    check("fault: CUDA fault aborts the whole run", p.returncode == 4 and v["verdict"].startswith("ABORTED-CUDA-FAULT"), f"rc={p.returncode} {v['verdict']}")
    # stage 2 gate
    p = subprocess.run(py + ["--stage", "2", "--out", str(workdir / "s2_nogate")], capture_output=True, text=True)
    check("gate: stage 2 refuses to start without STAGE1_PASS.json", p.returncode != 0 and "STAGE1_PASS" in (p.stderr + p.stdout), (p.stderr + p.stdout)[-200:])
    p, rows, s = go("stage2_shim", 2, ["--skip-stage1-check"])
    check("gate: stage 2 (shim test bypass) runs all subjects with complete rows", R.completeness_problems(rows, ["synthetic0", "synthetic1"], 3) == [], p.stderr[-300:])
    # stage-1 gate logic on fabricated CUDA rows
    S = ["s0"]
    rows = fab_rows(S, 3, dict(frozen=14.0, af=3.0, pile=3.0))
    summ = dict(warmup=dict(ok=True), memory={m: dict(vram_poll=dict(peak_vram_delta_mib=100.0)) for m in R.METHODS})
    g = R.stage1_gate(rows, dict(cuda_executed=True, reps=3, subjects=S), summ)
    check("gate logic: clean CUDA stage 1 passes", g["passed"], str(g))
    g = R.stage1_gate(rows, dict(cuda_executed=True, reps=3, subjects=S), dict(summ, memory={}))
    check("gate logic: missing VRAM probes block stage 2", not g["passed"])
    rows2 = fab_rows(S, 3, dict(frozen=14.0, af=3.0, pile=3.0), overrides={("s0", 1, "pile"): dict(status=R.RESOURCE_FAIL, t_total_s=None)})
    g = R.stage1_gate(rows2, dict(cuda_executed=True, reps=3, subjects=S), summ)
    check("gate logic: a PILE resource failure blocks stage 2", not g["passed"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=None); ap.add_argument("--cases", type=int, default=60)
    a = ap.parse_args()
    rules_tests()
    pile_v2_tests(a.cases)
    wd = a.out or Path(tempfile.mkdtemp(prefix="g5v2_"))
    wd.mkdir(parents=True, exist_ok=True)
    runner_tests(wd)
    n_fail = sum(not ok for _n, ok, _d in RES)
    print(f"\n{len(RES) - n_fail}/{len(RES)} tests passed")
    if a.out:
        (a.out / "test_g5_v2_results.json").write_text(json.dumps([dict(name=n, passed=ok, detail=d) for n, ok, d in RES], indent=1))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
