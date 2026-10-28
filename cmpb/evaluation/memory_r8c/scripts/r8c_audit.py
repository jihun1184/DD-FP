"""Read-only local evidence audit. Usage: python -B scripts/r8c_audit.py --root ..

Only writes under this R8-C directory; never imports production or launches CUDA.
The JSON contains recomputations from archived records, NOT new measurements.
"""
import argparse
import ast
import hashlib
import json
import statistics as st
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def readj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args(); root = a.root.resolve()
    base = root / 'NEUROCOMPUTING_V1_INTEGRITY_RECOVERY/20261013'
    run = root / 'v1/ddfp_r7_runs/r7_20261014_180628'
    stages = {'e1': 'attempt_20261014_182211', 'e2': 'attempt_20261014_182057', 'e6': 'attempt_20261016_024751'}
    out = {'kind': 'INDEPENDENT_RECOMPUTATION_OF_RECORDED_EVIDENCE', 'source_root': str(root), 'stages': {}, 'source_hashes': {}}
    # Manifest paths are relative to their original stage. Do not trust PASS labels alone.
    for stage, attempt in stages.items():
        d = run / stage / attempt
        checks = []
        for line in (d / 'ARTIFACT_SHA256.txt').read_text().splitlines():
            digest, rel = line.split(None, 1); p = d / rel.strip().lstrip('*')
            checks.append({'file': str(p), 'expected': digest, 'actual': sha(p) if p.is_file() else None})
        out['stages'][stage] = {'reported_verdict': readj(d / 'verdict.json')['verdict'], 'manifest_entries': len(checks),
                                'manifest_ok': all(c['expected'] == c['actual'] for c in checks), 'checks': checks}
    ep = run / 'e6' / stages['e6'] / 'e6/rows.jsonl'
    rows = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
    timed = [r for r in rows if r['rep'] >= 0]; mem = [r for r in rows if r['method'] == 'mem']
    subjects = sorted({r['subject'] for r in timed}); methods = ['gpu_k1', 'ddfp_k16_d1', 'seq_fp_cpu']
    expected = {(s, rep, m) for s in subjects for rep in range(3) for m in methods}
    counts = Counter((r['subject'], r['rep'], r['method']) for r in timed)
    z = {'sha256': sha(ep), 'n_total': len(rows), 'n_timed': len(timed), 'n_mem': len(mem), 'n_subjects': len(subjects),
         'complete_20x3x3': len(subjects) == 20 and set(counts) == expected and all(v == 1 for v in counts.values()),
         'cuda_records': all(r['backend'] == 'cupy' and r['cuda_executed'] is True for r in rows), 'methods': {}}
    means = {}
    for m in methods:
        rr = [r for r in timed if r['method'] == m]
        per = {s: st.mean(r['t_algorithm_s'] for r in rr if r['subject'] == s) for s in subjects}
        means[m] = per
        pooled = [st.median(r['mem_pool_total_mib'] for r in rr if r['subject'] == s) for s in subjects]
        z['methods'][m] = {'mean_subject_mean_s': st.mean(per.values()), 'sd_subject_mean_s': st.stdev(per.values()),
                           'median_subject_median_pool_mib': st.median(pooled), 'pool_range_mib': [min(pooled), max(pooled)],
                           'deterministic_all_subjects': all(len({r['out_sha256'] for r in rr if r['subject'] == s}) == 1 for s in subjects)}
        if m != 'seq_fp_cpu':
            z['methods'][m].update(violations=sum(r['global_violations'] for r in rr),
                all_global_pass=all(r['global_dwc_ok'] is True for r in rr),
                memory_pass={k: {'mean': st.mean(r[m][k] for r in mem), 'min': min(r[m][k] for r in mem), 'max': max(r[m][k] for r in mem)}
                             for k in ['peak_vram_delta_mib', 'baseline_device_used_mib', 'peak_device_used_mib', 'peak_rss_mib']},
                sampler_errors=[r[m]['sampler_error'] for r in mem if r[m]['sampler_error']])
    z['mean_paired_ratio_dd_over_k1'] = st.mean(means['ddfp_k16_d1'][s] / means['gpu_k1'][s] for s in subjects)
    z['ratio_of_means_dd_over_k1'] = st.mean(means['ddfp_k16_d1'].values()) / st.mean(means['gpu_k1'].values())
    z['R_star_counts'] = dict(Counter(r['R_star'] for r in timed if r['method'] == 'ddfp_k16_d1'))
    z['different_k1_dd_subjects'] = sum(next(r['out_sha256'] for r in timed if r['subject'] == s and r['method'] == 'gpu_k1') !=
                                         next(r['out_sha256'] for r in timed if r['subject'] == s and r['method'] == 'ddfp_k16_d1') for s in subjects)
    ranking = sorted(subjects, key=lambda s: (st.median(r['t_algorithm_s'] for r in timed if r['subject'] == s and r['method'] == 'gpu_k1'), s))
    z['selected_subjects_min_lower_median_max'] = [ranking[0], ranking[(len(ranking)-1)//2], ranking[-1]]
    k1, dd = (z['methods'][m]['median_subject_median_pool_mib'] for m in methods[:2])
    z['SC_M'] = {'status': 'FAIL', 'original_classifier_label': 'F4_MODEL_OUT_OF_BAND', 'k1_in_band': 1729 <= k1 <= 2689,
                 'k16_in_band': 1284 <= dd <= 1997, 'ratio_in_band': 1.15 <= k1/dd <= 1.60, 'ratio': k1/dd,
                 'measured_over_frozen': {'k1': k1/1921.36, 'k16': dd/1426.39}}
    e1 = run / 'e1' / stages['e1'] / 'e1/subjects'
    e1rows = [readj(p) for p in sorted(e1.glob('*.json')) if not p.name.endswith('.error.json')]
    z['e1_cross_hash_match'] = all(r['out_sha256'] == next(q for q in e1rows if q['subject'] == r['subject'])[
        'hashes' if r['method'] != 'ddfp_k16_d1' else 'ddfp'][{'gpu_k1': 'gpu_k1_out', 'seq_fp_cpu': 'seq_out', 'ddfp_k16_d1': 'out_sha256'}[r['method']]] for r in timed)
    z['e1_R_act_lower'] = {q['subject']: q['gpu_meta']['lower']['total_activation_rounds'] for q in e1rows}
    out['e6'] = z
    # Preserve original sources and human-readable numbered copies, including the frozen R8-A model.
    sources = [root / 'v1/DD-FP-R7/src/ddfp' / n for n in ['gpu_immersion.py', 'gpu_native_worklist.py', 'gpu_seed_aware_worklist.py']]
    sources += [root / 'v1/DD-FP-R7/scripts/ddfp/experiment_DDFP_all.py']
    sources += list((root / 'v1/DDFP_R7_GPU_SERVER_RUNNER_R0/r7').glob('*.py'))
    sources += list((base / 'R8B_SCALING_EXECUTION').glob('*.py'))
    sources += [base / 'R8B_SCALING_EXECUTION/run8.sh', base / 'R8B_SCALING_EXECUTION/PREREG_R8B_AMENDMENT_A1.md']
    sources += list((base / 'R8_SCALING_FEASIBILITY').glob('*'))
    for p in sources:
        if not p.is_file(): continue
        rel = p.relative_to(root); dst = HERE / 'frozen' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha(dst) != sha(p): raise ValueError(f'frozen snapshot differs; create a new audit version: {dst}')
        if not dst.exists(): dst.write_bytes(p.read_bytes())
        out['source_hashes'][str(rel)] = sha(p)
        if p.suffix == '.py':
            text = p.read_text(encoding='utf-8'); ast.parse(text)
            num = HERE / 'evidence/numbered' / rel.with_suffix('.txt'); num.parent.mkdir(parents=True, exist_ok=True)
            num.write_text('\n'.join(f'{i}: {l}' for i, l in enumerate(text.splitlines(), 1)) + '\n', encoding='utf-8')
    common = ast.parse((root / 'v1/DDFP_R7_GPU_SERVER_RUNNER_R0/r7/r7_common.py').read_text(encoding='utf-8'))
    exp = next(ast.literal_eval(n.value) for n in common.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EXPECTED' for t in n.targets))
    out['production_expected_matches'] = {p: sha(root / 'v1/DD-FP-R7' / p) == v for p, v in exp.items()}
    out['e6_script_hash_matches'] = {k: sha(root / 'v1/DDFP_R7_GPU_SERVER_RUNNER_R0/r7' / f) == readj(ep.parent / 'run_meta.json')[k]
                                     for k, f in [('script_sha256', 'r7_e6.py'), ('common_sha256', 'r7_common.py')]}
    out['zip_audit'] = []
    for p in [root / 'v1.zip', root / 'v1/DDFP_R7_GPU_SERVER_RUNNER_R0.zip', base / 'R8B_SCALING_EXECUTION/DDFP_R8B_EXECUTION_PACKAGE.zip', run / 'e6.zip']:
        if not p.is_file():
            out['zip_audit'].append({'path': str(p), 'status': 'MISSING_AT_AUDIT', 'sha256': None})
            continue
        with zipfile.ZipFile(p) as zz:
            names = zz.namelist()
            record = {'path': str(p), 'sha256': sha(p), 'entries': len(names), 'crc_error': zz.testzip(),
                                     'e6_rows_members': [n for n in names if n.endswith('e6/rows.jsonl')],
                                     'latest_e6_backup_present': any('attempt_20261016_024751/e6/rows.jsonl' in n for n in names)}
            if p.name == 'e6.zip':
                matches = []
                for n in names:
                    if n.endswith('/') or 'attempt_20261016_024751/' not in n: continue
                    suffix = n.split('attempt_20261016_024751/', 1)[1]
                    local = ep.parent.parent / suffix
                    digest = hashlib.sha256(zz.read(n)).hexdigest()
                    matches.append({'member': n, 'sha256': digest, 'local_match': local.is_file() and sha(local) == digest})
                record['latest_attempt_members'] = matches
                record['all_latest_members_equal_local'] = bool(matches) and all(x['local_match'] for x in matches)
            out['zip_audit'].append(record)
    # All input paths are read again at close to detect accidental modification during the audit.
    out['source_preservation_ok'] = all(sha(root / rel) == h for rel, h in out['source_hashes'].items())
    path = HERE / 'results/evidence_audit.json'; path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'file': str(path), 'counts': [z['n_timed'], z['n_mem']], 'SC_M': z['SC_M'], 'selected': z['selected_subjects_min_lower_median_max']}, indent=2))


if __name__ == '__main__':
    main()
