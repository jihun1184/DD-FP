#!/usr/bin/env python3
"""DD-FP v1 CPU-vs-GPU PARITY (P1.1 gate, item C). Standalone: needs only numpy (+ cupy for the real GPU run).
Scope: 3x3x5 counterexample + 49 fixed fixtures (fixtures.json, 50 total). NOT a benchmark; no N=20 / Full-N.
Per fixture and tie policy (lower, higher) compares CPU reference (vendor/cpu/seedref.py) vs GPU (vendor/gpu/*), BIT FOR BIT:
  out    : float32 output bits of every scheduler call           tie  : tie events, level transitions, direction/activation counts
  seed   : seed count/activations/raw-bits flag, is_seed, scheduled & injection level   state: discovered/processed/pending
  dwc    : (face, cube) violation counts of every output and of the IBI assembly
 e2e (free-running IBI, GPU feeds its own pins) AND per-solve (GPU re-solves the CPU-recorded inputs; isolates error propagation).
Exit codes: 0 all pass | 1 mismatch/exception | 2 environment/integrity problem. The window is never closed by this script.
Usage:  python run_gpu_parity.py [--backend cupy|stub|stub-mutant] [--selfcheck] [--only ID_PREFIX] [--no-pause]"""
import sys, os, json, time, hashlib, platform, argparse, traceback, copy
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; V = HERE / "vendor"
IGN_META = {"implementation", "implementation_version", "canonical_native_v1_1_sha256", "p2_s1_contract_zip_sha256",
            "unseeded_delegated_to_canonical", "bucket_count", "field_V_MIN", "field_V_MAX", "closure_rounds_by_level",
            "pending_bitset_nonzero_words_final"}
CATS = {"tie": {"equidistant_next_level_tie_events", "level_transitions", "schedule_direction_counts", "backward_scheduler_transition_count",
                "total_activation_rounds", "scheduler_transition_count", "tie_policy"},
        "seed": {"seed_count", "seed_activation_count", "seed_raw_bits_preserved", "l_inf_validated_integer"},
        "state": {"nan_count", "discovered_count", "processed_count", "discovered_equals_processed", "pending_total_final"}}
AUX_CAT = {"is_seed": "seed", "scheduled_level": "seed", "seed_injection_level": "seed", "seed_value": "seed",
           "discovered": "state", "processed": "state", "pending_count": "state"}

def sha(b): return hashlib.sha256(b).hexdigest()
def fsha(p): return sha(Path(p).read_bytes())
def bits(a): return np.ascontiguousarray(np.asarray(a, np.float32)).view(np.uint32)

def integrity():
    """Fail closed: every vendored file and the fixture list must match the recorded SHA-256."""
    bad = []
    for ln in (V / "VENDOR_SHA256.txt").read_text().splitlines():
        if ln.strip():
            h, rel = ln.split(None, 1); rel = rel.strip().lstrip("*")
            if fsha(V / rel) != h: bad.append(rel)
    fj = json.loads((HERE / "fixtures.json").read_text())
    if sha(json.dumps(fj["fixtures"], separators=(",", ":")).encode()) != fj["sha256_of_list"]: bad.append("fixtures.json")
    return bad, fj

def env_info(cp):
    e = dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(), machine=platform.machine())
    if cp is not None:
        try:
            d = cp.cuda.Device(); pr = cp.cuda.runtime.getDeviceProperties(d.id)
            e.update(cupy=cp.__version__, gpu=pr["name"].decode() if isinstance(pr["name"], bytes) else str(pr["name"]),
                     cc=d.compute_capability, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_driver=cp.cuda.runtime.driverGetVersion())
        except Exception as ex: e["gpu_query_error"] = repr(ex)
    return e

# ---------------------------------------------------------------- comparison
def cmp_call(c, g):
    """c, g = (out, meta, aux). Returns {category: [messages]} (empty dict == bit-identical in every category)."""
    bad = {}; add = lambda k, m: bad.setdefault(k, []).append(m)
    co, go = np.asarray(c[0]), np.asarray(g[0])
    if co.shape != go.shape: add("out", f"shape {co.shape} vs {go.shape}")
    else:
        d = np.argwhere(bits(co) != bits(go))
        if len(d): add("out", f"{len(d)} cells differ, first {tuple(map(int, d[0]))}: cpu={co[tuple(d[0])]} gpu={go[tuple(d[0])]}")
    for k in sorted((set(c[1]) & set(g[1])) - IGN_META):
        cat = next((n for n, s in CATS.items() if k in s), "state")
        if json.dumps(c[1][k], sort_keys=True, default=str) != json.dumps(g[1][k], sort_keys=True, default=str):
            add(cat, f"meta[{k}] cpu={str(c[1][k])[:80]} gpu={str(g[1][k])[:80]}")
    for k in sorted(set(c[2]) & set(g[2])):
        a, b = np.asarray(c[2][k]).astype(np.float64).ravel(), np.asarray(g[2][k]).astype(np.float64).ravel()
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True): add(AUX_CAT.get(k, "state"), f"aux[{k}] differs")
    return bad

def dwc(I, a): f, k = ND(a); return [int(f), int(k)]

def solve_cpu(lo, hi, l, s, tie): return SEEDED(lo, hi, l, s, tie)

# ---------------------------------------------------------------- fixture runners
def make_tile(fx):
    vol = np.array(fx["vol"], np.uint8); lo, hi, l = BUILD(vol); seeds = None
    if fx["pins"]:
        seeds = np.full(lo.shape, np.nan, np.float32)
        for x, y, z, v in fx["pins"]: lo[x, y, z] = hi[x, y, z] = seeds[x, y, z] = v
    return lo, hi, l, seeds

def run_tile(fx, tie, gpu):
    lo, hi, l, s = make_tile(fx); r = dict(kind="tile", tie=tie, n_solves=1, bad={}, exc=None)
    c = solve_cpu(lo.copy(), hi.copy(), l, None if s is None else s.copy(), tie)
    c2 = solve_cpu(lo.copy(), hi.copy(), l, None if s is None else s.copy(), tie)
    r["cpu_deterministic"] = not cmp_call(c, c2)
    g = gpu(lo.copy(), hi.copy(), l, None if s is None else s.copy(), tie)
    r["bad"] = cmp_call(c, g); r["dwc_cpu"], r["dwc_gpu"] = dwc(None, c[0]), dwc(None, g[0])
    if r["dwc_cpu"] != r["dwc_gpu"]: r["bad"].setdefault("dwc", []).append(f"{r['dwc_cpu']} vs {r['dwc_gpu']}")
    r["tie_events_cpu"] = int(c[1]["equidistant_next_level_tie_events"]); r["seeds"] = int(c[1]["seed_count"])
    r["cpu_digest"] = sha(bits(c[0]).tobytes()); return r

def run_ibi(fx, tie, gpu):
    vol = np.array(fx["vol"], np.uint8); K, d = fx["K"], fx["delta"]; r = dict(kind="ibi", tie=tie, bad={}, exc=None)
    def traced(fn, rec):
        def w(lo, hi, l, s, t):
            ci = (lo.copy(), hi.copy(), l, None if s is None else s.copy(), t); o = fn(lo, hi, l, s, t); rec.append((ci, o)); return o
        return w
    rc, rg = [], []
    orig = IBI.seed_aware_cpu_reference
    try:
        IBI.seed_aware_cpu_reference = traced(solve_cpu, rc); A = IBI.ibi(vol, K, d, 8, tie=tie)
        IBI.seed_aware_cpu_reference = traced(gpu, rg);       B = IBI.ibi(vol, K, d, 8, tie=tie)
    finally: IBI.seed_aware_cpu_reference = orig
    r.update(n_solves=len(rc), R_cpu=A["R"], R_gpu=B["R"], dwc_cpu=dwc(None, A["u"]), dwc_gpu=dwc(None, B["u"]),
             tie_events_cpu=sum(int(o[1]["equidistant_next_level_tie_events"]) for _, o in rc), cpu_digest=sha(bits(A["u"]).tobytes()))
    e2e = []
    if A["R"] != B["R"]: e2e.append(f"rounds {A['R']} vs {B['R']}")
    if len(rc) != len(rg): e2e.append(f"solve count {len(rc)} vs {len(rg)}")
    if not np.array_equal(bits(A["u"]), bits(B["u"])): e2e.append(f"assembly differs in {int((bits(A['u']) != bits(B['u'])).sum())} cells")
    if r["dwc_cpu"] != r["dwc_gpu"]: e2e.append(f"DWC {r['dwc_cpu']} vs {r['dwc_gpu']}")
    if e2e: r["bad"]["e2e"] = e2e
    # per-solve: GPU re-solves the CPU-recorded inputs (teacher forcing)
    for i, (ci, co) in enumerate(rc):
        go = gpu(ci[0].copy(), ci[1].copy(), ci[2], None if ci[3] is None else ci[3].copy(), ci[4])
        for k, m in cmp_call(co, go).items(): r["bad"].setdefault(k, []).append(f"solve#{i}: {m[0]}" + (f" (+{len(m)-1})" if len(m) > 1 else ""))
        dc, dg = dwc(None, co[0]), dwc(None, go[0])
        if dc != dg: r["bad"].setdefault("dwc", []).append(f"solve#{i}: {dc} vs {dg}")
    if fx["id"].startswith("F00"):
        cx = json.loads((V / "min_counterexample.json").read_text()); idx = tuple(np.array(cx["mismatch"]).T)
        lo, hi, l = BUILD(vol); fpl = SEEDED(lo, hi, l, None, tie)[0][1:-1, 1:-1, 1:-1]
        r["cx"] = dict(cpu_ddfp_matches_stored=bool(np.array_equal(A["u"][idx], cx["ddfp"])), gpu_ddfp_matches_stored=bool(np.array_equal(B["u"][idx], cx["ddfp"])),
                       plain_fp_matches_stored=bool(np.array_equal(fpl[idx], cx["fp"])), n_cells_ddfp_ne_plain_cpu=int((A["u"] != fpl).sum()),
                       n_cells_ddfp_ne_plain_gpu=int((B["u"] != fpl).sum()))
        if not (r["cx"]["cpu_ddfp_matches_stored"] and r["cx"]["gpu_ddfp_matches_stored"]): r["bad"].setdefault("e2e", []).append("stored counterexample not reproduced")
    return r

# ---------------------------------------------------------------- main
def pause(args):
    if not args.no_pause and sys.stdin and sys.stdin.isatty():
        try: input("\n[done] Press Enter to close this window ...")
        except Exception: pass

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backend", default="cupy", choices=["cupy", "stub", "stub-mutant"])
    ap.add_argument("--selfcheck", action="store_true", help="CPU only: integrity + determinism, writes cpu_digest")
    ap.add_argument("--only", default=""); ap.add_argument("--no-pause", action="store_true"); ap.add_argument("--out", default="")
    args = ap.parse_args(); t0 = time.time()
    bad, fj = integrity()
    if bad: print("INTEGRITY FAILURE (vendor/fixtures modified):", bad); pause(args); return 2
    sys.path[:0] = [str(V / "cpu"), str(V / "gpu")]
    global SEEDED, BUILD, ND, IBI
    import ibi_cpu as IBI; from cpu_fp import build_ispan_cpu as BUILD; from seedref import seed_aware_cpu_reference as SEEDED; from dwc import nd_violations as ND
    cp, gpu, why = None, None, ""
    if args.selfcheck or args.backend.startswith("stub"):
        n = [0]
        def gpu(lo, hi, l, s, t):
            o = SEEDED(lo, hi, l, s, t); n[0] += 1
            if args.backend == "stub-mutant" and n[0] % 17 == 5:           # injected fault: must be detected
                o = (o[0].copy(), o[1], o[2]); o[0][o[0].shape[0] // 2, 1, 1] += 1.0
            return o
    else:
        try:
            import cupy as cp; import gpu_seed_aware_worklist as G
            cp.cuda.Device(0).use(); cp.zeros(1).sum(); gpu = lambda lo, hi, l, s, t: G.seed_aware_boutry_bucket_gpu(cp, lo, hi, l, s, t)
        except Exception as ex:
            why = f"{type(ex).__name__}: {ex}"
    env = env_info(cp); env["backend"] = args.backend if gpu else "none"
    print(json.dumps(env, indent=1))
    if gpu is None:
        print("\nGPU backend unavailable ->", why, "\nInstall: see README_GPU_PARITY.md (pip install --user cupy-cuda12x). Nothing was run."); pause(args); return 2
    if cp is not None and "3080" not in env.get("gpu", ""): print("WARNING: device is not an RTX 3080 Ti:", env.get("gpu"))
    rows, nfail = [], 0
    for fx in fj["fixtures"]:
        if args.only and not fx["id"].startswith(args.only): continue
        for tie in ("lower", "higher"):
            tt = time.time()
            try: r = (run_tile if fx["kind"] == "tile" else run_ibi)(fx, tie, gpu)
            except Exception as ex: r = dict(kind=fx["kind"], tie=tie, bad={"exception": [repr(ex)]}, exc=traceback.format_exc())
            r.update(id=fx["id"], sec=round(time.time() - tt, 3)); r["pass"] = not r["bad"]; nfail += not r["pass"]; rows.append(r)
            print(f"{'PASS' if r['pass'] else 'FAIL'}  {fx['id']:<28} tie={tie:<6} solves={r.get('n_solves','?')} tieEv={r.get('tie_events_cpu','?')} "
                  f"dwc(cpu)={r.get('dwc_cpu','?')} dwc(gpu)={r.get('dwc_gpu','?')}" + ("" if r["pass"] else "  <- " + "; ".join(f"{k}:{v[0]}" for k, v in r["bad"].items())), flush=True)
    dig = sha("".join(r.get("cpu_digest", "") for r in rows).encode())
    exp = HERE / "expected_cpu_digest.txt"; note = ""
    if exp.exists() and not args.only:
        note = "CPU digest MATCHES reference run" if exp.read_text().split()[0] == dig else "CPU digest DIFFERS from reference run (numpy/platform drift?) - GPU-vs-CPU result above is still valid"
    summ = dict(env=env, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), runs=len(rows), failed=nfail, cpu_digest=dig, cpu_digest_note=note,
                cpu_nondeterministic=[r["id"] for r in rows if r.get("cpu_deterministic") is False], fixtures_sha256=fj["sha256_of_list"],
                vendor_sha256_file=fsha(V / "VENDOR_SHA256.txt"), script_sha256=fsha(__file__), seconds=round(time.time() - t0, 1), rows=rows)
    out = Path(args.out) if args.out else HERE / "results_gpu" / (time.strftime("%Y%m%d_%H%M%S") + f"_{summ['env']['backend']}")
    out.mkdir(parents=True, exist_ok=True); (out / "parity_report.json").write_text(json.dumps(summ, indent=1, default=str))
    if args.selfcheck and not args.only: (out / "cpu_digest.txt").write_text(dig + "\n")
    print(f"\n{'ALL PASS' if not nfail else 'FAILURES: ' + str(nfail)} of {len(rows)} runs | {note} | report: {out / 'parity_report.json'}")
    pause(args); return 1 if nfail else 0

if __name__ == "__main__":
    try: rc = main()
    except SystemExit as e: rc = e.code if isinstance(e.code, int) else 1
    except BaseException:
        traceback.print_exc(); print("\nINTERNAL ERROR (see traceback above). Please send the whole window text."); rc = 3
        try:
            if sys.stdin and sys.stdin.isatty(): input("\n[error] Press Enter to close ...")
        except Exception: pass
    sys.exit(rc)
