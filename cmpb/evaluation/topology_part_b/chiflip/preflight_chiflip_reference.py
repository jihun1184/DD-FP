from __future__ import annotations
import argparse, csv, glob
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--brats-dir',required=True)
    ap.add_argument('--ref-chi-csv',required=True)
    ap.add_argument('--expected-subjects',type=int,default=1251)
    a=ap.parse_args()
    brats=Path(a.brats_dir).resolve(); ref=Path(a.ref_chi_csv).resolve()
    files=sorted(glob.glob(str(brats/'*'/'*_flair.nii*')))
    ids=[Path(p).parent.name for p in files]
    if len(ids)!=a.expected_subjects or len(ids)!=len(set(ids)):
        raise SystemExit(f'BraTS discovery mismatch: files={len(ids)} unique_subjects={len(set(ids))} expected={a.expected_subjects}')
    with ref.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    rids=[r['subject_id'] for r in rows]
    if len(rids)!=a.expected_subjects or len(rids)!=len(set(rids)):
        raise SystemExit(f'Reference mapping mismatch: rows={len(rids)} unique_subjects={len(set(rids))} expected={a.expected_subjects}')
    missing=sorted(set(ids)-set(rids)); extra=sorted(set(rids)-set(ids))
    if missing or extra:
        raise SystemExit(f'Subject-set mismatch: missing_ref={missing[:10]} extra_ref={extra[:10]}')
    print(f'P6_K16_CHIFLIP_REFERENCE_PREFLIGHT_PASS subjects={len(ids)}')
if __name__=='__main__': main()
