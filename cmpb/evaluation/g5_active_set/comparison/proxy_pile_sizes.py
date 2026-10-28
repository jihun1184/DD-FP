import sys, json, time
import numpy as np
from pathlib import Path
R=Path('.').resolve()
sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R/'tests'));sys.path.insert(0,str(R/'comparison'))
import af_worklist as AF, generic_worklists as GW, harness as Hh
from fakecupy import FakeCupy
from scipy.ndimage import gaussian_filter
cp=FakeCupy(seed=5,threads=8)
F=AF.load_frozen()
res={}
for name,shape,sig,noise in (("smooth_s1.5",(72,72,48),1.5,0.0),("textured",(72,72,48),1.0,0.25),("rough",(72,72,48),0.6,0.0)):
    rng=np.random.default_rng(11)
    f=gaussian_filter(rng.standard_normal(shape),sig)
    f=(f-f.mean())/f.std()
    if noise: f=f+noise*rng.standard_normal(shape)
    # brain-like mask: ellipsoid, background 0
    g=np.stack(np.meshgrid(*[np.linspace(-1,1,s) for s in shape],indexing='ij'))
    mask=(g**2).sum(0)<0.8
    v=np.clip((f*30+110),1,255)*mask
    vol=np.rint(v).astype(np.uint8)
    lo,hi,l=Hh.CPU_FP.build_ispan_cpu(vol)
    t=time.time()
    out,meta,aux=GW.generic_pile_worklist(cp,lo,hi,l,"nearest-lower",256,frozen=F,trace=True)
    g_=meta['gw']; st=g_['stats']; N=g_['N']
    ps=np.array(g_['pile_sizes']); fs=np.array(g_['frontier_sizes'])
    res[name]=dict(N=N,visits=st['visits'],rounds=st['rounds'],max_pile=st['max_pile'],median_pile_at_visit=float(np.median(ps)),
        split_elements=st['split_elements'],gather_equivalent=st['visits']*N,traffic_ratio=st['split_elements']/(st['visits']*N),
        max_frontier=st['max_frontier'],pile_over_N_max=st['max_pile']/N, backward=meta['backward_scheduler_transition_count'], ties=meta['equidistant_next_level_tie_events'],
        seconds=round(time.time()-t,1))
    # equality with AF as sanity (AF on same case)
    af=AF.boutry_active_frontier(cp,lo,hi,l,"lower",256,None,frozen=F)
    res[name]['equal_AF_output']=bool(np.array_equal(af[0],out,equal_nan=True)); res[name]['equal_AF_rounds']=af[1]['total_activation_rounds']==meta['total_activation_rounds']
print(json.dumps(res,indent=1))
json.dump(res,open('/tmp/claude-0/-home-claude/55381a98-a13c-5dbf-b285-8961522ce03e/scratchpad/out/proxy_pile_sizes.json','w'),indent=1)
