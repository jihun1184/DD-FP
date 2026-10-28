"""R8-F1L statistics: tables (median/Q1-Q3/min/max), paired B/C ratios, H1-H4 verdicts with the F1K pre-registered thresholds (constants below are FROZEN; never edit after data).
  python -B scripts/r8f1l_stats.py summary --runs results/s0 results/s2 [...] --out stats.json
  python -B scripts/r8f1l_stats.py kstar --run results/s3run          # writes KSTAR.json (min median T_total of B and C at copies=1)
A cell = (input, K).  Timing statistics use ONLY purpose=timing records; memory/compare use the metadata pass; profile passes are never mixed into timing."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

H1_SUPPORT, H1_REJECT = 0.80, 0.95          # median T_total(B)/T_total(C)
H2_SUPPORT, H2_REJECT = 0.80, 0.95          # reserved-peak B/A
H3_SUPPORT_FRAC, H3_REJECT_RATIO, H3_F_UNI_MAX = 0.90, 2.0, 0.45   # F1K + evaluability precondition (freeze P-6)
H4_SUPPORT_REDUCTION = 0.50                  # (1 - near-seam |C-A| / |B-A|) >= 0.5


def q(v):
    v = np.asarray(v, float)
    return None if v.size == 0 else {'n': int(v.size), 'median': float(np.median(v)), 'q1': float(np.quantile(v, .25)), 'q3': float(np.quantile(v, .75)), 'min': float(v.min()), 'max': float(v.max())}


def load(runs):
    recs = []
    for r in runs:
        for f in sorted(Path(r, 'workers').glob('*/record.json')):
            d = json.loads(f.read_text(encoding='utf-8'))
            d['_run'], d['_id'] = str(r), f.parent.name
            recs.append(d)
    return recs


def cell(r):
    return (r['input'], r['method_id'], 1 if r['method_id'] == 'A' else r['K'])


def tables(recs):
    tim, met, defects, contended = defaultdict(list), {}, [], defaultdict(int)
    for r in recs:
        if r.get('status') != 'COMPLETED' or not r.get('validation', {}).get('ok'):
            defects.append({'id': r['_id'], 'status': r.get('status'), 'reasons': r.get('validation', {}).get('reasons') or r.get('error')})
            continue
        c = cell(r)
        if r['purpose'] == 'timing':
            tim[c].append(r)
            contended[c] += bool(r['gpu_state'].get('contended'))
        elif r['purpose'] == 'metadata':
            met[c] = r
    out = {}
    for c, rs in sorted(tim.items()):
        t = lambda f: [f(x) for x in rs]
        out['|'.join(map(str, c))] = {'T_total_direct_s': q(t(lambda x: x['t_total_s']['direct'])), 'T_alg_s': q(t(lambda x: x['t_total_s']['alg'])), 'T_cert_s': q(t(lambda x: x['t_total_s']['cert_wall'])),
                                      'event_over_wall': q(t(lambda x: x['cuda_event_envelope_s'] / x['algorithm_wall_s'])), 'n_contended': contended[c], 'cuda_executed': all(x['cuda_executed'] for x in rs)}
    return tim, met, defects, out


def med(c, tim):
    return float(np.median([x['t_total_s']['direct'] for x in tim[c]]))


def h1(tim, defects):
    cells = sorted({(i, k) for (i, m, k) in tim if m == 'B'} & {(i, k) for (i, m, k) in tim if m == 'C'})
    rows = []
    for i, k in cells:
        b, c = tim[(i, 'B', k)], tim[(i, 'C', k)]
        n = min(len(b), len(c))
        pair = [b[j]['t_total_s']['direct'] / c[j]['t_total_s']['direct'] for j in range(n)]              # same replicate index: shuffled within block, so a weak (not time-adjacent) pairing
        rows.append({'input': i, 'K': k, 'ratio_of_medians': med((i, 'B', k), tim) / med((i, 'C', k), tim), 'rep_pair_ratio': q(pair), 'alg_ratio': float(np.median([x['t_total_s']['alg'] for x in b]) / np.median([x['t_total_s']['alg'] for x in c]))})
    if not rows: return {'verdict': 'NOT MEASURED', 'cells': []}
    stat = float(np.median([r['ratio_of_medians'] for r in rows]))
    v = 'REJECTED' if (stat >= H1_REJECT or defects) else 'SUPPORTED' if stat <= H1_SUPPORT else 'UNDECIDED'
    return {'statistic_median_over_cells': stat, 'verdict': v, 'cells_B_faster': sum(r['ratio_of_medians'] < 1 for r in rows), 'n_cells': len(rows), 'range': [min(r['ratio_of_medians'] for r in rows), max(r['ratio_of_medians'] for r in rows)], 'cells': rows,
            'rule': f'support: median<= {H1_SUPPORT} and no defect; reject: >= {H1_REJECT} or any defect (F1K); aggregation over cells = median (freeze P-3)'}


def h1_secondary(tim, kstar):
    """Secondary, NOT the H1 test: each method at its OWN best K* (chosen on BraTS2021_00158 copies=1, held out for other inputs). Mixes the IBI effect with a different K, so it
    answers 'which configuration is cheapest', not 'what does removing IBI save'."""
    rows = []
    for i in sorted({i for (i, m, k) in tim}):
        kb, kc = kstar['B'], kstar['C']
        if (i, 'B', kb) in tim and (i, 'C', kc) in tim:
            rows.append({'input': i, 'K_B': kb, 'K_C': kc, 'ratio_best_B_over_best_C': med((i, 'B', kb), tim) / med((i, 'C', kc), tim), 'same_K': kb == kc})
    return {'role': 'secondary (best-configuration comparison; does not isolate the IBI effect)', 'cells': rows}


def h2(met):
    rows = []
    for (i, m, k), r in sorted(met.items()):
        if m != 'B' or 'memory' not in r: continue
        a = next((met[x] for x in met if x[0] == i and x[1] == 'A' and 'memory' in met[x]), None)
        if a is None: continue
        mb, ma = r['memory'], a['memory']
        rows.append({'input': i, 'K': k, 'reserved_ratio_B_over_A': mb['pool_reserved_peak_bytes'] / ma['pool_reserved_peak_bytes'], 'live_ratio_B_over_A': mb['pool_live_peak_bytes'] / ma['pool_live_peak_bytes'],
                     'device_sampled_ratio': mb['sampled_device_used_peak_bytes'] / ma['sampled_device_used_peak_bytes'],
                     'C_reserved_ratio_over_A': (met[(i, 'C', k)]['memory']['pool_reserved_peak_bytes'] / ma['pool_reserved_peak_bytes']) if (i, 'C', k) in met and 'memory' in met[(i, 'C', k)] else None})
    if not rows: return {'verdict': 'NOT MEASURED', 'cells': []}
    rr = [r['reserved_ratio_B_over_A'] for r in rows]
    v = 'SUPPORTED' if max(rr) <= H2_SUPPORT else 'REJECTED' if float(np.median(rr)) >= H2_REJECT else 'UNDECIDED'
    return {'verdict': v, 'max': max(rr), 'median': float(np.median(rr)), 'cells': rows,
            'rule': f'support: every cell <= {H2_SUPPORT}; reject: median >= {H2_REJECT} (F1K; aggregation freeze P-3). Primary metric = CuPy pool RESERVED peak.',
            'caveats': ['pool reserved/live are CuPy allocator metrics, not total RTX 4090 VRAM (CUDA context, non-pool allocations and other processes are outside)',
                        'device_sampled_ratio uses cudaMemGetInfo sampled every 10 ms and at each allocation: a LOWER BOUND of the true device peak (short spikes can be missed); it includes other processes',
                        'live, reserved and device-sampled ratios are all reported; the verdict uses reserved only']}


def h3_h4(compare_files):
    h3, h4 = [], []
    for f in compare_files:
        c = json.loads(Path(f).read_text(encoding='utf-8'))
        for name, p in c['pairs'].items():
            a, b = name.split('__')
            if a == 'A' and b[0] == 'B':
                if p['bit_identical']: v = 'NO_DIFFERENCE'
                elif p['uniform_expectation_d4'] > H3_F_UNI_MAX: v = 'NOT_EVALUABLE'
                elif p['frac_diff_within_d4'] >= H3_SUPPORT_FRAC: v = 'SUPPORTED'
                elif p['concentration_ratio_d4'] < H3_REJECT_RATIO: v = 'REJECTED'
                else: v = 'UNDECIDED'
                h3.append({'input': c['input'], 'pair': name, 'verdict': v, 'frac_diff_d4': p.get('frac_diff_within_d4'), 'f_uni': p.get('uniform_expectation_d4'), 'ratio': p.get('concentration_ratio_d4'), 'n_diff': p['n_diff']})
        for lab in c['labels']:
            if lab[0] == 'B':
                K = lab[1:]
                pb, pc, pbc = c['pairs'].get(f'A__B{K}'), c['pairs'].get(f'A__C{K}'), c['pairs'].get(f'B{K}__C{K}')
                if pb and pc and 'near_d4_mean_abs' in pb and 'near_d4_mean_abs' in pc:
                    red = None if pb['near_d4_mean_abs'] == 0 else 1 - pc['near_d4_mean_abs'] / pb['near_d4_mean_abs']
                    h4.append({'input': c['input'], 'K': int(K), 'B_near_mean_abs': pb['near_d4_mean_abs'], 'C_near_mean_abs': pc['near_d4_mean_abs'], 'reduction': red, 'B_near_max': pb['near_d4_max_abs'], 'C_near_max': pc['near_d4_max_abs'],
                               'n_diff_B_vs_C': pbc['n_diff'] if pbc else None, 'extra_round_on_B_measured': False})
    ev = [r for r in h3 if r['verdict'] in ('SUPPORTED', 'REJECTED', 'UNDECIDED')]
    s = {x['verdict'] for x in ev}
    h3v = 'NOT EVALUABLE' if not ev and h3 else 'NOT MEASURED' if not h3 else 'SUPPORTED' if s == {'SUPPORTED'} else 'REJECTED' if s == {'REJECTED'} else 'MIXED/UNDECIDED'
    red = [r['reduction'] for r in h4 if r['reduction'] is not None]
    # The pre-registered H4 contains a causal term, "effect of one additional IBI round applied to B" (indicator (i)), that no tool here measures: B-vs-C differences mix ALL rounds.
    # A verdict (SUPPORTED/REJECTED) is therefore withheld until every cell carries extra_round_on_B_measured=True.  Indicator (ii) is reported as an OBSERVATION only.
    full = bool(h4) and all(r['extra_round_on_B_measured'] for r in h4)
    verdict = 'NOT MEASURED' if not h4 else 'NOT FULLY EVALUATED' if not full else 'EVALUATION TOOL PRESENT (verdict rule to be applied by the frozen H4 rule)'
    obs = {'seam_deviation_reduction_C_vs_B_ii': {'n_cells': len(red), 'min': min(red) if red else None, 'median': float(np.median(red)) if red else None, 'max': max(red) if red else None,
                                                  'observed_ge_50pct_at_all_cells': bool(red) and min(red) >= H4_SUPPORT_REDUCTION, 'status': 'OBSERVATION (not a verdict)'},
           'additional_IBI_round_effect_i': 'NOT MEASURED (B-vs-C n_diff is a proxy that mixes all rounds; no causal reading)',
           'threshold_components_iii': 'see compare/*.json components (Delta comp B vs C)'}
    return ({'verdict': h3v, 'cells': h3, 'rule': f'support: >= {H3_SUPPORT_FRAC} of differing cells at d<=4; reject: concentration < {H3_REJECT_RATIO}x; evaluable only if f_uni <= {H3_F_UNI_MAX} (freeze P-6)'},
            {'verdict': verdict, 'observations': obs, 'cells': h4,
             'rule': f'(ii) pre-registered support: C cuts near-seam mean |dev from A| by >= {H4_SUPPORT_REDUCTION} vs B; (i) extra IBI round on B not implemented => no verdict'})


def summary(a):
    recs = load(a.runs)
    tim, met, defects, tab = tables(recs)
    cmp_files = sorted(f for r in a.runs for f in Path(r, 'compare').glob('*.json'))
    h3, h4 = h3_h4(cmp_files)
    memt = {'|'.join(map(str, c)): {k: v for k, v in r['memory'].items() if k.endswith('_bytes')} for c, r in sorted(met.items()) if 'memory' in r}
    sec = h1_secondary(tim, json.loads(Path(a.kstar).read_text())) if a.kstar else {'role': 'secondary', 'cells': [], 'status': 'NOT COMPUTED (no --kstar)'}
    out = {'n_records': len(recs), 'n_defect_or_incomplete': len(defects), 'defects': defects, 'timing': tab, 'memory_metadata_pass': memt, 'H1': h1(tim, defects), 'H1_secondary_best_config': sec, 'H2': h2(met), 'H3': h3, 'H4': h4,
           'note': 'GPU and CPU times are never combined; stub (cuda_executed=false) records must not be used as GPU evidence',
           'any_stub': any(not r.get('cuda_executed') for r in recs)}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps({k: (v['verdict'] if isinstance(v, dict) and 'verdict' in v else v) for k, v in out.items() if k in ('n_records', 'n_defect_or_incomplete', 'H1', 'H2', 'H3', 'H4', 'any_stub')}, indent=1))


def kstar(a):
    tim, _, defects, _ = tables(load([a.run]))
    if defects: raise SystemExit('defects present; K* not defined')
    res = {}
    for m in 'BC':
        c = {k: med((i, mm, k), tim) for (i, mm, k) in tim if mm == m and i.endswith(':1')}
        if not c: raise SystemExit(f'no copies=1 timing for {m}')
        res[m] = min(c, key=c.get)
    Path(a.run, 'KSTAR.json').write_text(json.dumps(res), encoding='utf-8')
    print(res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('summary'); s.add_argument('--runs', nargs='+', type=Path, required=True); s.add_argument('--out', type=Path, required=True); s.add_argument('--kstar', type=Path, help='KSTAR.json from the S3 run (enables the secondary best-configuration comparison)')
    k = sp.add_parser('kstar'); k.add_argument('--run', type=Path, required=True)
    a = ap.parse_args()
    try:
        summary(a) if a.cmd == 'summary' else kstar(a)
    except (OSError, KeyError, ValueError) as e:
        sys.exit(f'stats error: {type(e).__name__}: {e}')
