"""CPU-only unit tests (stdlib unittest).  python -B -m unittest discover -s tests -v   (run from the package root)"""
import copy
import json
import shutil
import uuid
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import r8f1l_compare as CMP      # noqa: E402
import r8f1l_lib as L            # noqa: E402
import r8f1l_runner as R         # noqa: E402
import r8f1l_stats as S          # noqa: E402

PY = [sys.executable, '-B']


def scratch():
    d = ROOT / 'results/_unittest' / uuid.uuid4().hex[:8]
    d.mkdir(parents=True)
    return d


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


class Planner(unittest.TestCase):
    def test_counts_and_unique_ids(self):
        for st, n in (('S0', 15), ('S2', 176), ('S3', 126), ('S4', 72)):
            js = R.make_plan(st)
            self.assertEqual(len(js), n, st)
            self.assertEqual(len({j['id'] for j in js}), n)
            self.assertTrue(all(j['K'] == 1 for j in js if j['method'] == 'A'))
            self.assertTrue(all(j['K'] >= 2 for j in js if j['method'] != 'A'))

    def test_budget_constants(self):
        self.assertLessEqual(sum(R.EST_H.values()), R.CAP_H)
        self.assertEqual((R.CAP_H, R.WALL_RULE, R.CAP_GB), (9.1, 1.5, 8.0))

    def test_gated_stages_refuse_without_gate(self):
        d = scratch()
        r = run(PY + ['scripts/r8f1l_runner.py', '--stage', 'S3', '--run', str(d / 'x'), '--config', 'tests/stub_config.json', '--backend', 'stub'])
        shutil.rmtree(d, ignore_errors=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Gate G-S2', r.stderr + r.stdout)


class Validator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpd = scratch()
        plan = [dict(id=f'U_{m}{K}_metadata_0', stage='S2', input=f'synth:randomwalk:12x12x30:64:{5 + i}', method=m, K=K, purpose='metadata', rep=0) for i, (m, K) in enumerate((('A', 1), ('B', 3), ('C', 3)))]   # one input per method: group size 1, so the saved output survives for re-validation
        pf = Path(str(cls.tmpd)) / 'p.json'; pf.write_text(json.dumps(plan))
        r = run(PY + ['scripts/r8f1l_runner.py', '--stage', 'S2', '--run', str(Path(str(cls.tmpd)) / 'run'), '--config', 'tests/stub_config.json', '--backend', 'stub', '--plan-file', str(pf)])
        assert r.returncode == 0, r.stderr
        cls.rec = {m: json.loads((Path(str(cls.tmpd)) / 'run/workers' / f'U_{m}_metadata_0/record.json').read_text()) for m in ('A1', 'B3', 'C3')}
        cls.dir = {m: Path(str(cls.tmpd)) / 'run/workers' / f'U_{m}_metadata_0' for m in cls.rec}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpd, ignore_errors=True)

    def ok(self, m, mut=None, stub=True):
        r = copy.deepcopy(self.rec[m])
        if mut: mut(r)
        return L.validate_record(r, self.dir[m], allow_stub=stub)['ok']

    def test_clean_stub_passes_but_never_as_gpu(self):
        for m in self.rec:
            self.assertTrue(self.ok(m))
            self.assertFalse(self.ok(m, stub=False), 'stub record must never validate as real CUDA')

    def test_mutations_are_rejected(self):
        muts = {
            'dwc violation': lambda r: r['verification'].update(global_violations=1, global_dwc_ok=False),
            'nan skipped': lambda r: r['verification'].update(nan_skipped=1),
            'original voxel changed': lambda r: r['array_check'].update(ok=False, original_voxel_differences=1),
            'interval violation': lambda r: r['array_check'].update(ok=False, original_interval_violations=1),
            'sources changed': lambda r: r.update(sources_unchanged=False),
            'bad hash': lambda r: r.update(out_sha256='zz'),
            'nonpositive time': lambda r: r.update(algorithm_wall_s=0),
        }
        for name, mut in muts.items():
            for m in self.rec:
                self.assertFalse(self.ok(m, mut), f'{name} not rejected for {m}')

    def test_method_specific(self):
        self.assertFalse(self.ok('B3', lambda r: r.update(R_star=1)), 'B must have R_star 0')
        self.assertFalse(self.ok('B3', lambda r: r.update(max_rounds=8)), 'B must be max_rounds 0')
        self.assertFalse(self.ok('C3', lambda r: r.update(max_rounds=0)))
        self.assertFalse(self.ok('C3', lambda r: r.update(R_star=8)), 'C must converge strictly before the cap')
        self.assertFalse(self.ok('B3', lambda r: r['verification'].update(seam_violations=1)))
        self.assertFalse(self.ok('A1', lambda r: r.update(K=2)))


class Compare(unittest.TestCase):
    def test_selftest(self):
        self.assertTrue(CMP.selftest()['pass'])


    def test_corner_mean_deviation_planted(self):
        import numpy as np
        v = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
        u = np.zeros((7, 7, 7), np.float32)
        for i in range(7):
            for j in range(7):
                for k in range(7):
                    cs = [v[a, b, d] for a in ([i // 2, (i + 1) // 2] if i % 2 else [i // 2]) for b in ([j // 2, (j + 1) // 2] if j % 2 else [j // 2]) for d in ([k // 2, (k + 1) // 2] if k % 2 else [k // 2])]
                    u[i, j, k] = np.mean(cs)
        u[1, 1, 1] += 1
        h = CMP.interval_position_hist(u, v)
        self.assertAlmostEqual(h['mean_abs_dev_from_corner_mean'] * h['non_original_cells'], 1.0, places=6)
        self.assertAlmostEqual(h['max_abs_dev_from_corner_mean'], 1.0, places=6)


class Stats(unittest.TestCase):
    def pair(self, fd, fu, ratio, nd=100, identical=False):
        return {'cells': 1000, 'n_diff': 0 if identical else nd, 'bit_identical': identical, 'frac_diff_within_d4': fd, 'uniform_expectation_d4': fu, 'concentration_ratio_d4': ratio}

    def h3(self, p):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'c.json'
            f.write_text(json.dumps({'input': 'x', 'labels': ['A', 'B8'], 'pairs': {'A__B8': p}}))
            return S.h3_h4([f])[0]['cells'][0]['verdict']

    def test_h3_rules(self):
        self.assertEqual(self.h3(self.pair(.95, .2, 4.75)), 'SUPPORTED')
        self.assertEqual(self.h3(self.pair(.30, .2, 1.5)), 'REJECTED')
        self.assertEqual(self.h3(self.pair(.60, .2, 3.0)), 'UNDECIDED')
        self.assertEqual(self.h3(self.pair(1., .5, 2.)), 'NOT_EVALUABLE')          # f_uni > 0.45: pre-registered rule cannot discriminate (freeze P-6)
        self.assertEqual(self.h3(self.pair(0, .2, None, identical=True)), 'NO_DIFFERENCE')

    def test_frozen_constants(self):
        self.assertEqual((S.H1_SUPPORT, S.H1_REJECT, S.H2_SUPPORT, S.H2_REJECT, S.H3_SUPPORT_FRAC, S.H3_REJECT_RATIO, S.H4_SUPPORT_REDUCTION), (.8, .95, .8, .95, .9, 2., .5))


class H4Budget(unittest.TestCase):
    def cmp(self, d, red_ok=True):
        f = Path(d) / 'c.json'
        pb = {'bit_identical': False, 'n_diff': 10, 'near_d4_mean_abs': 1.0, 'near_d4_max_abs': 4.0, 'frac_diff_within_d4': .5, 'uniform_expectation_d4': .2, 'concentration_ratio_d4': 2.5}
        pc = dict(pb, near_d4_mean_abs=.2 if red_ok else 1.0)
        f.write_text(json.dumps({'input': 'x', 'labels': ['A', 'B8', 'C8'], 'pairs': {'A__B8': pb, 'A__C8': pc, 'B8__C8': dict(pb)}}))
        return f

    def test_h4_never_gets_a_verdict_while_extra_round_unmeasured(self):
        for ok in (True, False):
            with tempfile.TemporaryDirectory() as d:
                h4 = S.h3_h4([self.cmp(d, ok)])[1]
                self.assertEqual(h4['verdict'], 'NOT FULLY EVALUATED')
                self.assertEqual(h4['observations']['seam_deviation_reduction_C_vs_B_ii']['observed_ge_50pct_at_all_cells'], ok)
                self.assertIn('NOT MEASURED', h4['observations']['additional_IBI_round_effect_i'])
        self.assertEqual(S.h3_h4([])[1]['verdict'], 'NOT MEASURED')

    def test_h2_reports_all_three_and_caveats(self):
        mem = lambda r, l, d: {'memory': {'pool_reserved_peak_bytes': r, 'pool_live_peak_bytes': l, 'sampled_device_used_peak_bytes': d}}
        met = {('x', 'A', 1): mem(100, 80, 150), ('x', 'B', 4): mem(60, 50, 120), ('x', 'C', 4): mem(70, 60, 130)}
        h = S.h2(met)
        self.assertEqual(h['verdict'], 'SUPPORTED')
        self.assertTrue({'reserved_ratio_B_over_A', 'live_ratio_B_over_A', 'device_sampled_ratio'} <= set(h['cells'][0]))
        self.assertTrue(any('LOWER BOUND' in c for c in h['caveats']) and any('not total' in c for c in h['caveats']))

    def test_s3b_same_k_for_b_and_c(self):
        for kb, kc in ((8, 16), (2, 128), (32, 32)):
            js = R.plan_S3b(kb, kc)
            for sid in ('BraTS2021_01363', 'BraTS2021_01628'):
                Kb = {j['K'] for j in js if j['input'].endswith(sid + ':1') and j['method'] == 'B'}
                Kc = {j['K'] for j in js if j['input'].endswith(sid + ':1') and j['method'] == 'C'}
                self.assertEqual(Kb, Kc)
                self.assertTrue({kb, kc} <= Kb)
        self.assertLessEqual(max(len(R.plan_S3b(a, b)) for a in R.KLADDER for b in R.KLADDER), 104)
        self.assertLessEqual(sum(R.EST_H.values()), R.CAP_H)

    def test_secondary_best_config_is_separate(self):
        rec = lambda t: [{'t_total_s': {'direct': t}}]
        tim = {('x', 'B', 8): rec(1.0), ('x', 'C', 16): rec(2.0), ('x', 'B', 16): rec(3.0), ('x', 'C', 8): rec(2.5)}
        h = S.h1_secondary(tim, {'B': 8, 'C': 16})
        self.assertAlmostEqual(h['cells'][0]['ratio_best_B_over_best_C'], .5)
        self.assertIn('does not isolate', h['role'])
        self.assertNotIn('verdict', h)

    def test_ledger_includes_prechecks_and_guards(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'ledger.json'
            self.assertEqual(L.ledger_spent(path=p), 0)
            L.ledger_add({'backend': 'stub', 'stage': 'PRE_s1', 'gpu_wall_h': 5}, path=p)
            self.assertEqual(L.ledger_spent(path=p), 0, 'stub never consumes hours')
            for st, h in (('PRE_b_equiv', .1), ('PRE_negctl', .01), ('PRE_timer', .01), ('PRE_s1', .4), ('S0', .2)):
                L.ledger_add({'backend': 'cupy', 'stage': st, 'gpu_wall_h': h}, path=p)
            self.assertAlmostEqual(L.ledger_spent('S0', p), .32)            # b_equiv + negctl + timer + S0 share one group
            self.assertAlmostEqual(L.ledger_spent('S1', p), .4)
            self.assertAlmostEqual(L.ledger_spent(path=p), .72)
            L.ledger_add({'backend': 'cupy', 'stage': 'PRE_timer', 'gpu_wall_h': .5}, path=p)           # S0 group now 0.82 > 1.5 x 0.5
            with self.assertRaises(SystemExit): L.budget_guard('PRE_negctl', path=p)
            L.ledger_add({'backend': 'cupy', 'stage': 'S2', 'gpu_wall_h': 9.0}, path=p)
            with self.assertRaises(SystemExit): L.budget_guard('S4', path=p)


class Gate(unittest.TestCase):
    def test_stub_evidence_never_passes(self):
        res = ROOT / 'results'
        runs = sorted(res.glob('_unittest*')) and []
        d = scratch()
        pf = d / 'p.json'
        pf.write_text(json.dumps([dict(id=f'G_{m}_{p}_0', stage='S2', input='synth:randomwalk:12x12x30:64:5', method=m, K=1 if m == 'A' else 3, purpose=p, rep=0) for m in 'ABC' for p in ('timing', 'metadata')]))
        rr = run(PY + ['scripts/r8f1l_runner.py', '--stage', 'S2', '--run', str(d / 'run'), '--config', 'tests/stub_config.json', '--backend', 'stub', '--plan-file', str(pf)])
        self.assertEqual(rr.returncode, 0, rr.stderr)
        runs = [d / 'run']
        if not runs: self.skipTest('no stub run available')
        r = run(PY + ['scripts/r8f1l_gate.py', '--s0', str(runs[-1]), '--s2', str(runs[-1]), '--b-equiv', 'results/stub_b_equiv.json', '--negctl', 'results/stub_negctl.json',
                      '--timer', 'results/stub_timer.json', '--s1', 'results/stub_s1.json', '--out', str(Path(tempfile.gettempdir()) / 'g.json')])
        self.assertEqual(r.returncode, 4)
        self.assertIn('"decision": "HOLD"', r.stdout)


if __name__ == '__main__':
    unittest.main()
