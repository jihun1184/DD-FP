#!/usr/bin/env python3
"""
gate_verdict.py -- fail-closed verdict logic for the active-frontier (AF) gate, plus the gather-share aggregation rule.

Replaces the verdict block at the end of ``run_gpu_gate.py`` as executed for G3 (SHA ae2d38dd...).  Two defects of that block:
  (D1) ``perf = "GO" if (... and (not extra_mem or max(extra_mem) <= 100))``  -> GO was possible with NO memory measurement
       (``--no-mem``, a sampler that returned nothing, or a subject without a "memory" entry).
  (D2) ``perf`` was computed from the speedups alone; ``correctness_ok == False`` did not influence it, so a run could print
       correctness=FAIL together with performance=GO.
Rules implemented here (everything not explicitly satisfied is NOT a GO):
  correctness  PASS        every subject has correct_default_cap is True and correct_small_cap is True, the number of subjects equals the
                           declared n_subjects, the global flag agrees, and (when the rows file is given) every timed run reproduces
                           the reference output hash and round count.
               FAIL        any of the above is False.      INCOMPLETE   data missing / counts inconsistent (and no explicit failure).
  performance  NOT-ELIGIBLE   correctness is not PASS (no speed/memory claim is derived from a run whose outputs are not verified)
               NOT-MEASURED   run was not on CUDA (shim)
               INCOMPLETE     a speedup is missing / non-finite / <= 0
               PILOT          fewer than MIN_SUBJECTS_FOR_GO subjects (a GO is a cohort statement; n=1 can only be a pilot)
               NO-GO          median speedup < 1.3
               GO             median >= 2.0 and min >= 1.3 and EVERY subject has a finite extra-memory measurement and max <= 100 MiB
               HOLD           otherwise (this includes: memory missing for >= 1 subject)
Gather-share rule (criterion "gather > 40 % of AF time", MECH_GATE/A0-A1 report s.3, which named no cohort aggregation):
  per subject   g_s = gather GPU time (CUDA events, kernels af_gather + the mode-0 af_finalize launch) / total wall time of the SAME
                intrusive breakdown pass of the AF run   (summary.subjects[s].breakdown.af: prof.gpu_ms.gather / total_s)
  cohort        PRIMARY  = median over subjects of g_s;  the 40 % criterion is "met" iff PRIMARY > 0.40.
                REPORTED ALONGSIDE (not decisive): number of subjects with g_s > 0.40, time-weighted pooled share, and a sensitivity
                with the headline (non-intrusive) AF median time as denominator.
  The primary rule is fixed here BEFORE the G4 data exist and is not to be re-chosen after seeing them.
"""
from __future__ import annotations

import math
import statistics as st

VERDICT_CODE_VERSION = "gate_verdict v2 (fail-closed)"
MEDIAN_GO, MIN_GO, MEDIAN_NOGO, MEM_GO_MIB, MIN_SUBJECTS_FOR_GO = 2.0, 1.3, 1.3, 100.0, 20
GATHER_SHARE_LIMIT = 0.40
RULE_TEXT = ("GO: n>=20 & median>=2.0 & min>=1.3 & memory measured for every subject & max extra<=100MiB; NO-GO: median<1.3; "
             "NOT-ELIGIBLE if correctness!=PASS; otherwise HOLD")


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ------------------------------------------------------------------------------------------------------------------------
def timed_rows_consistency(summary: dict, rows: list[dict]) -> list[str]:
    """Every timed run (rows.jsonl) must carry the reference hash and round count of its subject; reps*2 rows per subject."""
    bad = []
    reps = summary.get("reps")
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r.get("subject"), []).append(r)
    for sid, rec in summary.get("subjects", {}).items():
        rr = by.get(sid, [])
        if _num(reps) and len(rr) != 2 * int(reps):
            bad.append(f"{sid}: {len(rr)} timed rows, expected {2 * int(reps)}")
        for r in rr:
            if r.get("out_sha256") != rec.get("output_sha256"):
                bad.append(f"{sid}: rep {r.get('rep')} {r.get('method')} output hash differs from reference")
            if r.get("rounds") != rec.get("frozen_rounds"):
                bad.append(f"{sid}: rep {r.get('rep')} {r.get('method')} round count {r.get('rounds')} != {rec.get('frozen_rounds')}")
        for m in ("frozen", "af"):
            ts = [r["t_total_s"] for r in sorted(rr, key=lambda r: r.get("rep", 0)) if r.get("method") == m]
            if ts != list(rec.get("times", {}).get(m, [])):
                bad.append(f"{sid}: {m} times in rows.jsonl differ from summary")
    extra = set(by) - set(summary.get("subjects", {}))
    if extra:
        bad.append(f"rows for unknown subjects: {sorted(extra)[:5]}")
    return bad


def correctness_status(summary: dict, rows: list[dict] | None = None) -> tuple[str, list[str]]:
    subs = summary.get("subjects")
    if not isinstance(subs, dict) or not subs:
        return "INCOMPLETE", ["no per-subject records"]
    fail, incomplete = [], []
    if summary.get("n_subjects") != len(subs):
        incomplete.append(f"n_subjects={summary.get('n_subjects')} but {len(subs)} subject records")
    for sid, r in subs.items():
        for key in ("correct_default_cap", "correct_small_cap"):
            if key not in r:
                incomplete.append(f"{sid}: {key} missing")
            elif r[key] is not True:
                fail.append(f"{sid}: {key}={r[key]!r}")
    if summary.get("correctness_ok") is not True and not fail:
        fail.append(f"global correctness_ok={summary.get('correctness_ok')!r} although no per-subject failure is recorded")
    if rows is not None:
        fail += timed_rows_consistency(summary, rows)
    if fail:
        return "FAIL", fail + incomplete
    if incomplete:
        return "INCOMPLETE", incomplete
    return "PASS", []


def decide(summary: dict, rows: list[dict] | None = None) -> dict:
    corr, creasons = correctness_status(summary, rows)
    out = dict(code=VERDICT_CODE_VERSION, rule=RULE_TEXT, correctness=corr, correctness_reasons=creasons[:20],
               n_subjects=len(summary.get("subjects", {})), reasons=[])
    subs = summary.get("subjects", {})
    sp = [r.get("speedup") for r in subs.values()]
    mem = {sid: (r.get("memory") or {}).get("extra_device_mib_af_minus_frozen") for sid, r in subs.items()}
    good_sp = [x for x in sp if _num(x) and x > 0]
    out["median_speedup"] = st.median(good_sp) if good_sp else None
    out["min_speedup"] = min(good_sp) if good_sp else None
    good_mem = [x for x in mem.values() if _num(x)]
    out["max_extra_device_mib"] = max(good_mem) if good_mem else None
    out["memory_missing_subjects"] = sorted(s for s, x in mem.items() if not _num(x))
    if corr != "PASS":
        out["performance"] = "NOT-ELIGIBLE"
        out["reasons"].append(f"correctness={corr}: no performance verdict is derived from an unverified run")
        return out
    if not summary.get("cuda_executed"):
        out["performance"] = "NOT-MEASURED"
        out["reasons"].append("run was not executed on CUDA")
        return out
    if len(good_sp) != len(sp) or not sp:
        out["performance"] = "INCOMPLETE"
        out["reasons"].append("speedup missing / non-finite / non-positive for at least one subject")
        return out
    med, mn = out["median_speedup"], out["min_speedup"]
    if len(sp) < MIN_SUBJECTS_FOR_GO:
        out["performance"] = "PILOT"
        out["reasons"].append(f"n={len(sp)} < {MIN_SUBJECTS_FOR_GO}: a cohort GO cannot be assigned (median {med:.3f}, min {mn:.3f})")
        return out
    if med < MEDIAN_NOGO:
        out["performance"] = "NO-GO"
        out["reasons"].append(f"median speedup {med:.3f} < {MEDIAN_NOGO}")
        return out
    if out["memory_missing_subjects"]:
        out["performance"] = "HOLD"
        out["reasons"].append(f"extra-memory measurement missing for {len(out['memory_missing_subjects'])} subject(s): GO impossible")
        return out
    if med >= MEDIAN_GO and mn >= MIN_GO and out["max_extra_device_mib"] <= MEM_GO_MIB:
        out["performance"] = "GO"
        return out
    out["performance"] = "HOLD"
    if med < MEDIAN_GO:
        out["reasons"].append(f"median {med:.3f} < {MEDIAN_GO}")
    if mn < MIN_GO:
        out["reasons"].append(f"min {mn:.3f} < {MIN_GO}")
    if out["max_extra_device_mib"] > MEM_GO_MIB:
        out["reasons"].append(f"max extra memory {out['max_extra_device_mib']:.1f} MiB > {MEM_GO_MIB}")
    return out


# ------------------------------------------------------------------------------------------------------------------------
def gather_share_stats(summary: dict) -> dict:
    """Primary rule + the alternatives that were considered (see module docstring).  Pure arithmetic on recorded numbers."""
    per, per_headline, miss = {}, {}, []
    tg = tt = 0.0
    for sid, r in summary.get("subjects", {}).items():
        try:
            b = r["breakdown"]["af"]
            g = float(b["prof"]["gpu_ms"]["gather"]) / 1e3
            tot = float(b["total_s"])
            per[sid] = g / tot
            tg += g
            tt += tot
            per_headline[sid] = g / float(r["median_s"]["af"])
        except (KeyError, TypeError, ZeroDivisionError, ValueError):
            miss.append(sid)
    v = list(per.values())
    vh = list(per_headline.values())
    return dict(
        rule="g_s = prof.gpu_ms.gather / breakdown.af.total_s (same intrusive pass); PRIMARY = median_s g_s > 0.40",
        n=len(v), missing=miss,
        primary_median=st.median(v) if v else None,
        primary_criterion_met=(st.median(v) > GATHER_SHARE_LIMIT) if v else None,
        n_subjects_over_limit=sum(1 for x in v if x > GATHER_SHARE_LIMIT),
        per_subject_min=min(v) if v else None, per_subject_max=max(v) if v else None,
        pooled_time_weighted=(tg / tt) if tt else None,
        sensitivity_headline_denominator_median=st.median(vh) if vh else None,
        sensitivity_headline_denominator_n_over=sum(1 for x in vh if x > GATHER_SHARE_LIMIT),
        per_subject=per,
    )
