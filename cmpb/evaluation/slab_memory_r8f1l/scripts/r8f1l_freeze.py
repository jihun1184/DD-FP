"""Freeze manifest: SHA-256 of every code/config/protocol file, of the registered job plans, and of the frozen constants + decision record.
  python -B scripts/r8f1l_freeze.py write        # once, BEFORE any GPU data exists (refuses to overwrite)
  python -B scripts/r8f1l_freeze.py check        # run_server.sh calls this first; any drift => exit 3 and no GPU work starts"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
MANIFEST = ROOT / 'FREEZE_MANIFEST.json'

DECISIONS = {
    'basis': 'User message 2026-10-18 22:25 JST: three pre-run corrections requested, P-1..P-7 approvals and code/config SHA to be re-frozen, "no further research-design decision is needed". '
             'Defaults in R8F1L_PROTOCOL_FREEZE.md section 8 are recorded as approved by that statement; P-7 default = keep the registered randomwalk inputs. If this reading is wrong, do not run.',
    'P-1': 'S2 timing n=5; S2 estimate 1.7 h', 'P-2': 'one profile pass per S2 configuration; never mixed into timing', 'P-3': 'cell aggregation rules as in stats constants/docstrings; verdicts per stage family, no pooling across S0/S2/S3/S3b/S4',
    'P-4': 'S3b: B and C run at the SAME K (union of both +-1 ladders around the S3 copies=1 K*); MAIN = same-K comparison; SECONDARY = each method at its own K* (does not isolate IBI)',
    'P-5': '1 Hz nvidia-smi contention poll', 'P-6': 'H3 evaluable only if f_uni <= 0.45', 'P-7': 'registered randomwalk inputs kept; saturation disclosed',
    'A-1': 'H4: verdict withheld (NOT FULLY EVALUATED); seam-deviation reduction reported as observation; extra-IBI-round effect NOT MEASURED',
    'A-2': 'prechecks b_equiv/negctl/timer (group S0) and s1 (group S1) write to the same GPU-hour ledger; gate criterion 6 requires them on the ledger',
    'A-3': 'H2 caveats printed with the result: pool metrics are not VRAM; device sample is a lower bound (10 ms + allocation hooks); live/reserved/device ratios all reported',
    'A-4': 'EST S3b raised 0.75 -> 1.4 h (worst case 104 jobs); sum of stage estimates 7.6 h <= 9.1 h cap',
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def files():
    pats = ['scripts/*.py', 'tests/*.py', 'tests/*.json', 'config/*', 'run_server.sh', 'vendor/VENDOR_SHA256SUMS.txt', 'R8F1L_PROTOCOL_FREEZE.md']
    out = sorted({p for pat in pats for p in ROOT.glob(pat) if p.is_file()})
    return {str(p.relative_to(ROOT)): sha(p) for p in out}


def build():
    import r8f1l_runner as R
    import r8f1l_stats as S
    import r8f1l_lib as L
    plans = {st: hashlib.sha256(json.dumps(R.make_plan(st), sort_keys=True).encode()).hexdigest() for st in ('S0', 'S2', 'S3', 'S4')}
    plans['S3b_worst_case_jobs'] = max(len(R.plan_S3b(a, b)) for a in R.KLADDER for b in R.KLADDER)
    consts = {'EST_H': L.EST_H, 'CAP_H': L.CAP_H, 'WALL_RULE': L.WALL_RULE, 'CAP_GB': L.CAP_GB,
              'H1': [S.H1_SUPPORT, S.H1_REJECT], 'H2': [S.H2_SUPPORT, S.H2_REJECT], 'H3': [S.H3_SUPPORT_FRAC, S.H3_REJECT_RATIO, S.H3_F_UNI_MAX], 'H4_reduction': S.H4_SUPPORT_REDUCTION}
    return {'schema': 'R8F1L-FREEZE-1', 'files': files(), 'plans': plans, 'constants': consts, 'decisions': DECISIONS}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'write':
        if MANIFEST.exists(): sys.exit('manifest exists; refusing to overwrite (a re-freeze needs a new version directory)')
        MANIFEST.write_text(json.dumps(build(), indent=1, sort_keys=True), encoding='utf-8')
        print('written', sha(MANIFEST))
    elif cmd == 'check':
        want, got = json.loads(MANIFEST.read_text(encoding='utf-8')), build()
        bad = [k for k in sorted(set(want['files']) | set(got['files'])) if want['files'].get(k) != got['files'].get(k)]
        bad += [f'plan:{k}' for k in want['plans'] if want['plans'][k] != got['plans'].get(k)]
        bad += ['constants'] if want['constants'] != got['constants'] else []
        bad += ['decisions'] if want['decisions'] != got['decisions'] else []
        print(json.dumps({'ok': not bad, 'drift': bad, 'manifest_sha256': sha(MANIFEST)}, indent=1))
        sys.exit(0 if not bad else 3)
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    try:
        main()
    except (OSError, KeyError, ValueError, ImportError) as e:
        sys.exit(f'freeze error: {type(e).__name__}: {e}')
