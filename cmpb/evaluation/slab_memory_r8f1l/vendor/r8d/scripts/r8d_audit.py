"""D0: safely ingest R8-C return tar, recompute rows, and recover source bytes.

python -B scripts/r8d_audit.py --archive PATH --zip PATH --search-root PATH
No solver or untrusted archive Python is executed. Archive paths and symlinks
are checked before manual extraction; original evidence is never overwritten.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics as st
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def dump(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def ingest(archive, dest):
    dest.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive) as t:
        for m in t:
            path=PurePosixPath(m.name)
            if '\\' in m.name or path.is_absolute() or '..' in path.parts or any(':' in s for s in path.parts):
                raise ValueError('unsafe archive path')
            target=dest.joinpath(*path.parts)
            if not target.resolve().is_relative_to(dest.resolve()): raise ValueError('archive path escaped destination')
            if m.isdir(): target.mkdir(parents=True,exist_ok=True)
            elif m.isfile():
                if m.size > 64*2**20: raise ValueError('summary archive unexpectedly large member')
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f: f.write(t.extractfile(m).read())
            else: raise ValueError('links/devices refused')
    roots=list(dest.glob('*/run_config.json'))
    if len(roots)!=1: raise ValueError('ambiguous return root')
    return roots[0].parent


def audit(d, search):
    read=lambda p: json.loads(p.read_text(encoding='utf-8'))
    manifest=read(d/'run_config.json'); analysis=read(d/'analysis.json')
    records=[(p,read(p)) for p in sorted(d.glob('workers/*/record.json'))]
    groups={}; errors=[]; hash_checks=[]
    for manifest_path in sorted(d.rglob('SHA256SUMS.txt')):
        for line in manifest_path.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            expected,rel=line.split(None,1); p=(manifest_path.parent/rel.strip()).resolve()
            if not p.is_relative_to(d.resolve()): raise ValueError('manifest traversal')
            status='MISSING' if not p.is_file() else 'MATCH' if sha(p)==expected else 'MISMATCH'
            hash_checks.append({'manifest':str(manifest_path.relative_to(d)), 'path':str(p.relative_to(d)), 'status':status})
    sources=[]
    for original,expected in manifest['source_hashes'].items():
        name=PurePosixPath(original).name
        if not name.startswith('r8c_') or not name.endswith('.py'): continue
        candidates=[]
        for base in search:
            for p in base.rglob(name):
                if ROOT in p.parents: continue
                if sha(p)==expected: candidates.append(p)
        # Search known source bundle ZIPs when a matching loose file is absent.
        payload=None; recovered_from=None
        if candidates: payload=candidates[0].read_bytes(); recovered_from=str(candidates[0])
        else:
            for base in search:
                for zpath in base.glob('R8C*.zip'):
                    try:
                        with zipfile.ZipFile(zpath) as z:
                            for n in z.namelist():
                                if PurePosixPath(n).name==name:
                                    b=z.read(n)
                                    if hashlib.sha256(b).hexdigest()==expected:
                                        payload=b; recovered_from=str(zpath)+'!'+n; break
                    except zipfile.BadZipFile: continue
                    if payload is not None: break
                if payload is not None: break
        rec={'recorded_path':original,'sha256':expected,'recovered_from':recovered_from,'matched':payload is not None}
        if payload is not None:
            p=ROOT/'vendor/r8c/scripts'/name; p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists() and p.read_bytes()!=payload: raise ValueError('refuse vendor overwrite')
            p.write_bytes(payload); rec['preserved_path']=p.relative_to(ROOT).as_posix()
        sources.append(rec)
    for p,r in records:
        if r['purpose'] not in ('timing','metadata'): continue
        groups.setdefault((r['subject'],r['copies'],r['method']),[]).append((p,r))
        k=1 if r['method']=='gpu_k1' else 16 if r['method']=='ddfp_k16' else 16*r['copies']
        if r['K']!=k: errors.append(str(p)+': wrong K')
        if r['status']!='COMPLETED' or r['backend']!='cupy' or r['cuda_executed'] is not True:
            errors.append(str(p)+': not completed CUDA')
        if not (r['verification'].get('global_violations')==0 and r['verification'].get('nan_skipped')==0 and r['verification'].get('global_dwc_ok') is True and r['array_check'].get('ok') is True):
            errors.append(str(p)+': recorded correctness failure')
        if sha(p.parent/'environment.json')!=r['environment_sha256']: errors.append(str(p)+': environment hash')
        if read(p.parent/'environment.json')['sources']!=manifest['source_hashes']: errors.append(str(p)+': sources differ')
    rows=[]; ratios=[]; repeat_ratios=[]
    for key,rr in sorted(groups.items()):
        times=[r for _,r in rr if r['purpose']=='timing']; meta=[r for _,r in rr if r['purpose']=='metadata']
        if sorted(r['rep'] for r in times)!=[0,1,2] or len(meta)!=1: errors.append(str(key)+': missing repetitions'); continue
        if len({r['out_sha256'] for _,r in rr})!=1: errors.append(str(key)+': nondeterministic hash')
        q=meta[0]; mm=q['memory']; solves=q['metadata']['solves']
        med=st.median(r['t_algorithm_s'] for r in times)
        old=next(a for a in analysis['rows'] if (a['subject'],a['copies'],a['method'])==key)
        if not math.isclose(med,old['t_median_s'],abs_tol=1e-10): errors.append(str(key)+': analysis median mismatch')
        if old['source_sha256']!=sha(d/old['source']): errors.append(str(key)+': analysis source hash mismatch')
        rows.append({'subject':key[0],'copies':key[1],'method':key[2],'K':q['K'],
            'time_median_s':med,'time_mean_s':st.mean(r['t_algorithm_s'] for r in times),
            'time_repetitions_s':[r['t_algorithm_s'] for r in sorted(times,key=lambda x:x['rep'])],
            'pool_live_peak_gib':mm['pool_used_peak_bytes']/2**30,'pool_reserved_peak_gib':mm['pool_total_peak_bytes']/2**30,
            'R_star':q.get('R_star'),'n_solves':len(solves),'R_act_total':sum(s['total_activation_rounds'] for s in solves),
            'activation_cell_visits':sum(s['total_activation_rounds']*s['padded_cells'] for s in solves),
            'slab_padded_cells_sum':sum(s['padded_cells'] for s in solves),
            'metadata_to_timing_ratio':q['t_algorithm_s']/med,
            'out_sha256':q['out_sha256'],'source':str((d/old['source']).relative_to(ROOT))})
    for r in rows:
        if r['K']==1: continue
        b=next(q for q in rows if (q['subject'],q['copies'],q['K'])==(r['subject'],r['copies'],1))
        ratios.append({'subject':r['subject'],'copies':r['copies'],'method':r['method'],'ratio':r['time_median_s']/b['time_median_s']})
        for i,(v,w) in enumerate(zip(r['time_repetitions_s'],b['time_repetitions_s'])):
            repeat_ratios.append({'subject':r['subject'],'copies':r['copies'],'method':r['method'],'rep_index':i,'ratio':v/w,
                                  'scope':'matched repetition index, not simultaneous paired acquisition'})
    gm=[]
    for m in (1,2,4):
        for method in ('ddfp_k16','ddfp_k16m'):
            vals=[r['ratio'] for r in ratios if r['copies']==m and r['method']==('ddfp_k16' if m==1 else method)]
            if len(vals)!=3: errors.append('missing subject ratio group'); continue
            gm.append({'copies':m,'policy':method,'geomean_dd_over_gpu':math.exp(st.mean(math.log(v) for v in vals))})
    matched=sum(h['status']=='MATCH' for h in hash_checks); missing=[h for h in hash_checks if h['status']=='MISSING']
    mismatches=[h for h in hash_checks if h['status']=='MISMATCH']
    expected={(s,m,method) for s in manifest['subjects'] for m in (1,2,4) for method in ('gpu_k1','ddfp_k16','ddfp_k16m') if not(m==1 and method=='ddfp_k16m')}
    if set(groups)!=expected: errors.append('coverage differs')
    if mismatches: errors.append('manifest mismatch')
    if not all(s['matched'] for s in sources): errors.append('unresolved source bytes')
    return {'status':'RECOMPUTED_FROM_RETURN_RECORDS' if not errors else 'AUDIT_ERRORS','errors':errors,
        'record_count':len(records),'primary_count':sum(len(v) for v in groups.values()),'source_recovery':sources,
        'manifest_matches':matched,'manifest_missing':missing,'manifest_mismatches':mismatches,'manifest_checks':hash_checks,
        'rows':rows,'subject_ratios':ratios,'rep_index_ratios':repeat_ratios,'geomeans':gm,
        'recorded_validation':read(d/'validation.json'),'recorded_scientific_assessment':analysis['scientific_assessment'],
        'output_reverification':'NOT EXECUTED: summary return excludes output.npy; recorded correctness/hash assertions only',
        'metadata_note':'metadata passes synchronize/checkpoint and sample; their times are not baseline timing'}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--zip',type=Path); ap.add_argument('--search-root',type=Path,action='append',required=True)
    a=ap.parse_args(); dest=ROOT/'evidence/r8c_return'
    if dest.exists():
        previous=ROOT/'results/d0_audit.json'
        if not previous.is_file() or json.loads(previous.read_text(encoding='utf-8'))['archive']['sha256']!=sha(a.archive):
            raise ValueError('existing extraction fingerprint differs/absent; preserve it and use a new package')
        root=next(dest.glob('*/run_config.json')).parent
    else: root=ingest(a.archive,dest)
    result=audit(root,a.search_root); result['archive']={'path':str(a.archive),'sha256':sha(a.archive)}
    if a.zip:
        info={'path':str(a.zip),'size_bytes':a.zip.stat().st_size,'sha256':sha(a.zip)}
        try:
            with zipfile.ZipFile(a.zip) as z: info.update(status='READABLE',bad_crc_member=z.testzip())
        except zipfile.BadZipFile as e: info.update(status='UNREADABLE_BAD_ZIP',error=str(e))
        result['zip']=info
    dump(ROOT/'results/d0_audit.json',result)
    for name,rr in [('r8c_recomputed_rows',result['rows']),('r8c_subject_ratios',result['subject_ratios']),('r8c_rep_index_ratios',result['rep_index_ratios'])]:
        with (ROOT/'results'/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
    dump(ROOT/'config/vendor_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'vendor').rglob('*.py'))})
    print(json.dumps({k:result[k] for k in ('status','errors','record_count','manifest_matches','geomeans')},indent=2))


if __name__=='__main__': main()
