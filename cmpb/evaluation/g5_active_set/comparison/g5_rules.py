"""
g5_rules.py -- PRE-REGISTERED decision rules for the G5 comparison (frozen K=1 vs AF vs FP-aware generic pile worklist).

Pure Python / NumPy, no GPU.  Written and unit-tested BEFORE any G5 GPU number exists (the v1 rule in run_generic_gpu.py was never exercised on a GPU).
Changes against v1 (comparison/run_generic_gpu.py):
  * the label "AF-NECESSARY" is removed: a failing generic implementation is never read as a property of the algorithm class;
  * every run has exactly one status: OK | CORRECTNESS_FAIL | RESOURCE_FAIL | IMPL_ERROR | CUDA_FAULT, assigned per method, never mixed;
  * speed is judged only when all three methods are valid; ratios use a practical-equivalence margin and a bootstrap CI over subjects;
  * mixed per-subject outcomes have their own verdict.
The numbers below are fixed here; the runner copies them into summary.json together with the SHA-256 of this file.
"""
from __future__ import annotations

import statistics as st
from typing import Any

import numpy as np

RULES_VERSION = "G5_RULES_V2.0"
METHODS = ("frozen", "af", "pile")
OK, CORRECTNESS_FAIL, RESOURCE_FAIL, IMPL_ERROR, CUDA_FAULT = "OK", "CORRECTNESS_FAIL", "RESOURCE_FAIL", "IMPL_ERROR", "CUDA_FAULT"

MARGIN = 1.10            # practical-equivalence margin on r = t_AF / t_PILE (per-subject medians)
BOOT_B = 10_000          # bootstrap resamples over subjects
BOOT_SEED = 20261023
CI_PCT = (2.5, 97.5)
MIN_SUBJECTS = 20        # decision stage: the same 20 BraTS volumes as G3
MIN_REPS = 3
NOISE_RANGE = 0.10       # (max-min)/median of a cell's timed reps
NOISE_CELL_FRACTION = 0.20   # more than this fraction of noisy cells => timing judged too noisy to rank methods
ORDER_EFFECT = 0.03      # mean relative deviation between execution positions that is reported as a warning
G3_SPEEDUP_DRIFT = 0.10  # AF/frozen median speed-up vs G3 (warning only)


class GenericResourceError(RuntimeError):
    """capacity (list/pile) exceeded in the generic worklist -- a CONFIGURATION limit, not a semantic failure."""


class GenericInvariantError(RuntimeError):
    """an internal invariant of the generic worklist failed (pending-count mismatch, non-empty final state, safety limit) -- an implementation defect."""


def classify_exception(e: BaseException) -> str:
    n = type(e).__name__
    msg = str(e).lower()
    if isinstance(e, GenericResourceError) or n in ("OutOfMemoryError", "MemoryError") or "out of memory" in msg:
        return RESOURCE_FAIL
    if n in ("CUDARuntimeError", "CUDADriverError") or "illegal memory access" in msg or "unspecified launch failure" in msg:
        return CUDA_FAULT
    return IMPL_ERROR


def bootstrap_median_ci(vals, B: int = BOOT_B, seed: int = BOOT_SEED, pct=CI_PCT):
    v = np.asarray(vals, dtype=float)
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(B, v.size))
    meds = np.median(v[idx], axis=1)
    lo, hi = np.percentile(meds, pct)
    return (float(lo), float(hi))


def cells(rows):
    """{(subject, method): [row, ...]} over timed rows only."""
    out: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r.get("phase", "timed") == "timed":
            out.setdefault((r["subject"], r["method"]), []).append(r)
    return out


def completeness_problems(rows, subjects, reps: int, methods=METHODS) -> list[str]:
    """exactly one timed row for every (subject, rep, method); nothing extra."""
    seen: dict[tuple[str, int, str], int] = {}
    for r in rows:
        if r.get("phase", "timed") != "timed":
            continue
        k = (r["subject"], int(r["rep"]), r["method"])
        seen[k] = seen.get(k, 0) + 1
    bad = []
    for s in subjects:
        for rep in range(reps):
            for m in methods:
                c = seen.pop((s, rep, m), 0)
                if c != 1:
                    bad.append(f"{s}/rep{rep}/{m}: {c} rows")
    bad += [f"unexpected row {k}" for k in seen]
    return bad


def status_counts(rows) -> dict[str, dict[str, int]]:
    out = {m: {} for m in METHODS}
    for r in rows:  # warm-up, check, timed and memory runs count for failure classification; the intrusive decomposition ("profile") pass never does
        if r.get("phase") == "profile":
            continue
        d = out[r["method"]]
        d[r["status"]] = d.get(r["status"], 0) + 1
    return out


def per_subject_medians(rows, subjects) -> dict[str, dict[str, float]]:
    c = cells(rows)
    res = {}
    for s in subjects:
        res[s] = {}
        for m in METHODS:
            t = [r["t_total_s"] for r in c.get((s, m), []) if r["status"] == OK and r.get("t_total_s") is not None]
            if t:
                res[s][m] = st.median(t)
    return res


def noise_report(rows, subjects) -> dict[str, Any]:
    c = cells(rows)
    n_cells = n_noisy = 0
    worst = 0.0
    for s in subjects:
        for m in METHODS:
            t = [r["t_total_s"] for r in c.get((s, m), []) if r["status"] == OK and r.get("t_total_s") is not None]
            if len(t) >= 2:
                rng = (max(t) - min(t)) / st.median(t)
                n_cells += 1; n_noisy += int(rng > NOISE_RANGE); worst = max(worst, rng)
    return dict(cells=n_cells, noisy_cells=n_noisy, noisy_fraction=(n_noisy / n_cells if n_cells else float("nan")), worst_range=worst)


def order_effect(rows, subjects) -> dict[str, Any]:
    """mean of t / (cell median) per execution position, per method; flags |deviation| > ORDER_EFFECT."""
    c = cells(rows)
    acc: dict[tuple[str, int], list[float]] = {}
    for (s, m), rs in c.items():
        t = [r["t_total_s"] for r in rs if r["status"] == OK and r.get("t_total_s") is not None]
        if len(t) < 2:
            continue
        med = st.median(t)
        for r in rs:
            if r["status"] == OK and r.get("t_total_s") is not None:
                acc.setdefault((m, int(r["order_position"])), []).append(r["t_total_s"] / med)
    table = {f"{m}@pos{p}": float(np.mean(v)) for (m, p), v in sorted(acc.items())}
    worst = max((abs(x - 1.0) for x in table.values()), default=0.0)
    return dict(mean_rel_time_by_position=table, worst_deviation=worst, warning=bool(worst > ORDER_EFFECT))


def speed_summary(rows, subjects) -> dict[str, Any]:
    med = per_subject_medians(rows, subjects)
    r = {s: med[s]["af"] / med[s]["pile"] for s in subjects if "af" in med[s] and "pile" in med[s]}
    vals = list(r.values())
    out: dict[str, Any] = dict(per_subject_r_af_over_pile=r, n=len(vals))
    if vals:
        out.update(median_r=float(np.median(vals)), min_r=min(vals), max_r=max(vals),
                   pile_faster_subjects=sum(v >= MARGIN for v in vals), af_faster_subjects=sum(v <= 1 / MARGIN for v in vals),
                   tied_subjects=sum(1 / MARGIN < v < MARGIN for v in vals))
        out["ci95_median_r"] = bootstrap_median_ci(vals)
    sp = {m: [med[s]["frozen"] / med[s][m] for s in subjects if "frozen" in med[s] and m in med[s]] for m in ("af", "pile")}
    for m, v in sp.items():
        if v:
            out[f"median_speedup_{m}_vs_frozen"] = float(np.median(v)); out[f"ci95_speedup_{m}_vs_frozen"] = bootstrap_median_ci(v)
    return out


def decide(rows, meta: dict[str, Any]) -> dict[str, Any]:
    """meta: stage, cuda_executed, reps, subjects (list run), expected_subjects (G3 list or None), methods_ok_to_time (bool)."""
    subjects = list(meta["subjects"]); reps = int(meta["reps"])
    V: dict[str, Any] = dict(rules_version=RULES_VERSION, stage=meta.get("stage"), notes=[])
    counts = status_counts(rows)
    V["status_counts"] = counts
    if not meta.get("cuda_executed"):
        V["verdict"] = "NOT-MEASURED (shim/CPU run; plumbing only)"; return V
    if any(CUDA_FAULT in counts[m] for m in METHODS):
        V["verdict"] = "ABORTED-CUDA-FAULT (diagnose: environment vs generic-pile kernel; no research inference)"; return V
    if any(k != OK for k in counts["frozen"]):
        V["verdict"] = "INVALID-REFERENCE-FAILURE (frozen run failed; nothing can be compared)"; return V
    if any(k != OK for k in counts["af"]):
        V["verdict"] = "INVALID-AF-REGRESSION (AF disagrees with frozen or failed; contradicts G3/G4; investigate before any G5 inference)"; return V
    pile_bad = {k: v for k, v in counts["pile"].items() if k != OK}
    if CORRECTNESS_FAIL in pile_bad or IMPL_ERROR in pile_bad:
        V["verdict"] = "PILE-IMPLEMENTATION-FAILURE (diagnose root cause; NOT evidence that AF-specific design is necessary)"
        V["required_diagnosis"] = ["kernel defect (race/atomics/indexing)", "capacity or configuration", "FP-contract mis-specification in the generic design",
                                    "harness/timing-scope defect"]
        V["pile_failures"] = pile_bad; return V
    if RESOURCE_FAIL in pile_bad:
        V["verdict"] = "PILE-RESOURCE-LIMITED (configuration-specific capacity/OOM; semantics not tested on the failed subjects; no speed verdict)"
        V["pile_failures"] = pile_bad; V["speed_descriptive"] = speed_summary(rows, subjects); return V
    prob = completeness_problems(rows, subjects, reps)
    if prob:
        V["verdict"] = "INVALID-RUN (missing or duplicate timed rows)"; V["problems"] = prob[:20]; return V
    S = speed_summary(rows, subjects); V["speed"] = S
    V["noise"] = noise_report(rows, subjects); V["order_effect"] = order_effect(rows, subjects)
    exp = meta.get("expected_subjects")
    if len(subjects) < MIN_SUBJECTS or reps < MIN_REPS or (exp is not None and list(subjects) != list(exp)):
        V["verdict"] = "NOT-DECIDABLE (needs the same 20 G3 subjects and >=3 reps); stage-level correctness/resource result only"; return V
    if V["noise"]["noisy_fraction"] > NOISE_CELL_FRACTION:
        V["verdict"] = "INDETERMINATE-NOISY (timing dispersion too large to rank methods)"; return V
    lo, hi = S["ci95_median_r"]
    if lo >= MARGIN:
        V["verdict"] = "PILE-FASTER"
    elif hi <= 1 / MARGIN:
        V["verdict"] = "AF-FASTER"
    elif lo >= 1 / MARGIN and hi <= MARGIN:
        V["verdict"] = "EQUIVALENT"
    elif S["pile_faster_subjects"] >= 3 and S["af_faster_subjects"] >= 3:
        V["verdict"] = "MIXED-SUBJECT-DEPENDENT"
    else:
        V["verdict"] = "INDETERMINATE (CI straddles the equivalence margin)"
    if V["order_effect"]["warning"]:
        V["notes"].append("execution-position effect above 3%: inspect order_effect before relying on the ranking")
    return V


def stage1_gate(rows, meta: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Stage 1 (one BraTS subject) must PASS before stage 2 may run.  PASS means: valid run, all three methods OK in every rep, resource & memory probes present."""
    reasons = []
    if not meta.get("cuda_executed"):
        reasons.append("not a CUDA run")
    subjects = list(meta["subjects"])
    if len(subjects) != 1:
        reasons.append("stage 1 must contain exactly one subject")
    if completeness_problems(rows, subjects, int(meta["reps"])):
        reasons.append("rows incomplete")
    counts = status_counts(rows)
    for m in METHODS:
        if any(k != OK for k in counts[m]):
            reasons.append(f"{m}: {counts[m]}")
    if int(meta["reps"]) < MIN_REPS:
        reasons.append("reps < 3")
    if not summary.get("warmup", {}).get("ok"):
        reasons.append("warm-up check failed")
    mem = summary.get("memory", {})
    for m in METHODS:
        e = mem.get(m, {})
        if not (e.get("vram_poll", {}).get("peak_vram_delta_mib") is not None or e.get("project_sampler")):
            reasons.append(f"no VRAM measurement for {m}")
    return dict(passed=not reasons, reasons=reasons, rules_version=RULES_VERSION)
