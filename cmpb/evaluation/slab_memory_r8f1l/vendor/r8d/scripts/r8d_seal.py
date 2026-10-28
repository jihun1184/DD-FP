"""Create or verify the complete deliverable manifest; excludes mutable caches."""
import argparse
from r8d_common import ROOT,sha,seal


def verify():
    bad=[]
    for line in (ROOT/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        h,rel=line.split(None,1); p=(ROOT/rel.strip()).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or sha(p)!=h: bad.append(rel)
    if bad: raise ValueError('missing/changed files: '+str(bad))
    print('SHA256 manifest verified')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--verify',action='store_true'); a=p.parse_args()
    verify() if a.verify else seal(ROOT)
