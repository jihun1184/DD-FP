"""Gate G-S2 (brief sec.5): six criteria -> PASS | HOLD | FAIL.  PASS writes <s3-run>/GATE_S2_PASS.json (the only thing that unlocks S3/S3b/S4).
  python -B scripts/r8f1l_gate.py --s0 results/s0 --s2 results/s2 --b-equiv results/gpu_b_equiv.json --negctl results/gpu_negctl.json --timer results/gpu_timer.json --s1 results/gpu_s1.json --s3-run results/s3
FAIL = a correctness counterexample (any preserved record violating items 1-3 / seam / B,C invariants).  HOLD = anything else missing or insufficient.  Stub (CPU) evidence can never PASS."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import r8f1l_lib as L          # noqa: E402  (budget constants + the single ledger)
from r8f1l_lib import CAP_H, WALL_RULE, CAP_GB, EST_H   # noqa: E402
CORRECTNESS = ('items 1-2', 'item 3', 'seam check', 'B must report', 'C convergence', 'FP NaN', 'solve coverage')


def rj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def records(run):
    return [(f.parent.name, rj(f)) for f in sorted(Path(run, 'workers').glob('*/record.json'))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ('s0', 's2'): ap.add_argument('--' + k, type=Path, required=True)
    for k in ('b-equiv', 'negctl', 'timer', 's1'): ap.add_argument('--' + k, type=Path, required=True)
    ap.add_argument('--s3-run', type=Path)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/GATE_S2_DECISION.json')
    a = ap.parse_args()
    crit, notes, counter = {}, [], []
    recs = records(a.s0) + records(a.s2)
    pre = {k: rj(getattr(a, k.replace('-', '_'))) for k in ('b-equiv', 'negctl', 'timer', 's1')}
    p0 = rj(ROOT / 'evidence/p0_b_semantics.json')['summary']
    real = bool(recs) and all(r.get('cuda_executed') is True and r.get('backend') == 'cupy' for _, r in recs) and all(p.get('cuda_executed') is True for p in pre.values())
    notes.append(f'{len(recs)} records; all real CUDA = {real}')
    for i, r in recs:                                                                             # FAIL candidates: preserved correctness violations
        why = r.get('validation', {}).get('reasons', [])
        if r.get('status') in ('COMPLETED', 'VALIDATION_FAILED') and any(w.startswith(CORRECTNESS) or any(c in w for c in CORRECTNESS) for w in why):
            counter.append({'record': i, 'reasons': why})
    done = [(i, r) for i, r in recs if r.get('status') == 'COMPLETED' and r.get('validation', {}).get('ok')]
    # 1 Round-0 spec
    crit['1_B_matches_round0_spec'] = bool(p0['n_fail_cases'] == 0 and p0['n_cases'] >= 100 and pre['b-equiv']['pass'] and pre['b-equiv']['n'] >= 100 and pre['b-equiv']['cuda_executed'])
    # 2 invariants (negative controls, S1, every record validated, A/C regression hashes)
    reg = [r['regression'] for _, r in done if r.get('regression', {}).get('expected')]
    crit['2_invariants_kept'] = bool(recs and len(done) == len(recs) and pre['negctl']['pass'] and pre['s1']['pass'] and pre['s1']['n'] >= 200 and reg and all(x['ok'] for x in reg) and not counter)
    # 3 kernel time actually instrumented
    gpu_meta = [r for _, r in done if r['purpose'] == 'metadata']
    ev = [r['cuda_event_envelope_s'] / r['algorithm_wall_s'] for _, r in done if r['purpose'] == 'timing']
    crit['3_gpu_kernel_instrumented'] = bool(real and pre['timer']['pass'] and gpu_meta and all(r.get('cert_kernel', {}).get('kernel_event_s', 0) > 0 for r in gpu_meta) and ev and abs(np.median(ev) - 1) <= .02)
    # 4 completeness: per (input, method, K) >=3 timing, 1 metadata w/ memory + transfers + (A,B,C all present for each synthetic input)
    cells = {}
    for _, r in done: cells.setdefault((r['input'], r['method_id'], r['K']), []).append(r)
    miss = []
    for c, rs in cells.items():
        t = [r for r in rs if r['purpose'] == 'timing']; m = [r for r in rs if r['purpose'] == 'metadata']
        if c[0] == 'smoke' or rs[0]['stage'] == 'S0': continue                                     # S0 cells are covered by criteria 2/5
        if len(t) < 5 or not m or 'memory' not in m[0] or 'transfers' not in m[0]: miss.append(c)
    inputs = {c[0] for c in cells if c[0] != 'smoke' and cells[c][0]['stage'] == 'S2'}
    for i in inputs:
        for meth in 'ABC':
            if not any(c[0] == i and c[1] == meth for c in cells): miss.append((i, meth, 'absent'))
    crit['4_time_memory_complete'] = bool(done and inputs and not miss)
    if miss: notes.append(f'incomplete cells: {miss[:10]}')
    # 5 reproducibility: one output hash per cell across passes; S0 A copies=1 CV < 5 %
    nd = [c for c, rs in cells.items() if len({r['out_sha256'] for r in rs}) != 1]
    a_t = [r['t_total_s']['direct'] for _, r in done if r['stage'] == 'S0' and r['method_id'] == 'A' and r['purpose'] == 'timing' and r['input'].startswith('brats')]
    cv = float(np.std(a_t, ddof=1) / np.mean(a_t)) if len(a_t) >= 3 else None
    crit['5_reproducible'] = bool(done and not nd and cv is not None and cv < .05)
    notes.append(f'S0 A timing CV = {cv}; nondeterministic cells = {nd}')
    # 6 budget
    led = L.ledger_read()
    spent = L.ledger_spent()
    gb = sum(f.stat().st_size for f in (ROOT / 'results').rglob('*') if f.is_file()) / 2 ** 30
    stages = {e['stage'] for e in led}
    pre_ok = {'PRE_b_equiv', 'PRE_negctl', 'PRE_timer', 'PRE_s1', 'S0', 'S2'} <= stages               # prechecks must be on the ledger, not only the stages
    over = {g: L.ledger_spent(g) for g in EST_H if L.ledger_spent(g) > EST_H[g] * WALL_RULE}
    crit['6_within_budget'] = bool(led and pre_ok and spent <= CAP_H and gb <= CAP_GB and not over)
    notes.append(f'ledger spent {spent:.2f} h incl. prechecks (cap {CAP_H}); prechecks+stages on ledger = {pre_ok}; storage {gb:.2f} GB (cap {CAP_GB}); groups over 1.5x: {over}')
    verdict = 'FAIL' if counter else 'PASS' if all(crit.values()) and real else 'HOLD'
    out = {'decision': verdict, 'criteria': crit, 'counterexamples': counter, 'notes': notes, 'real_cuda_evidence': real}
    a.out.write_text(json.dumps(out, indent=1), encoding='utf-8')
    if verdict == 'PASS' and a.s3_run:
        a.s3_run.mkdir(parents=True, exist_ok=True)
        (a.s3_run / 'GATE_S2_PASS.json').write_text(json.dumps(out), encoding='utf-8')
    print(json.dumps(out, indent=1))
    return 0 if verdict == 'PASS' else 4 if verdict == 'HOLD' else 5


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, KeyError, ValueError) as e:
        sys.exit(f'gate error (treated as HOLD): {type(e).__name__}: {e}')
