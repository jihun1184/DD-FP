"""Reproduce F1 from D0 records only, after reports/Figure_Layout_Plans.md.

Requires matplotlib/numpy. Emits vector SVG and 300 dpi PNG. No phase timing,
unmeasured K-sweep values, interpolation optimum or causal arrow is invented.
"""
import json
import math
import statistics as st
from pathlib import Path
from r8d_common import ROOT,readj,writej


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fixed,scaled,neutral='#BF7B22','#147D82','#69737C'
    a=readj(ROOT/'results/d0_audit.json'); data=readj(ROOT/'results/cost_features.json')['rows']
    out=ROOT/'results/figures'; out.mkdir(exist_ok=True)
    fig=plt.figure(figsize=(7.0,5.1))
    gs=fig.add_gridspec(2,3,width_ratios=[1.0,1.05,1.45],height_ratios=[1,1],wspace=.43,hspace=.64)
    ax=fig.add_subplot(gs[:,0]); ax.set_title('(a) Core and halo',loc='left',fontweight='bold')
    for m,y in [(1,5),(2,3.5),(4,2)]:
        D=155*m
        for K,shift,col,label in [(16,.22,fixed,'K=16'),(16*m,-.22,scaled,f'K={16*m}')]:
            boundaries=[i*(D//K)+min(i,D%K) for i in range(K+1)]
            ax.add_patch(Rectangle((0,y+shift-.12),D/620,.24,facecolor=col,alpha=.15,edgecolor='none'))
            for z in boundaries:
                ax.plot([z/620,z/620],[y+shift-.12,y+shift+.12],color=col,lw=.45)
            ax.text(D/620+.02,y+shift,label,va='center',fontsize=7,color=col)
        ax.text(-.04,y,f'm={m}\nD={D}',ha='right',va='center',fontsize=7)
    ax.set_xlim(-.02,1.22); ax.set_ylim(-.5,5.65); ax.axis('off')
    ax.text(0,1.25,'Max. core = ceil(D/K)',fontsize=7)
    ax.text(0,.98,'fixed: 10, 20, 39',fontsize=7)
    ax.text(0,.72,'scaled: 10, 10, 10',fontsize=7)
    # Local inset preserves halo semantics using neutral hatch, policy color only for core.
    for x,col in [(0,fixed),(.44,fixed)]:
        ax.add_patch(Rectangle((x,-.04),.44,.3,facecolor=col,alpha=.15,edgecolor=col,lw=.8))
    ax.plot([.44,.44],[-.08,.30],color=neutral,lw=1)
    ax.add_patch(Rectangle((.36,-.08),.16,.38,facecolor='none',edgecolor=neutral,hatch='////',lw=.7))
    ax.text(.44,.44,'1-plane halo per side',ha='center',fontsize=7)
    ax.text(0,-.39,r'$\delta=1:\ \sum_j d_j=D+2(K-1)$',fontsize=7)
    acts=fig.add_subplot(gs[0,1]); work=fig.add_subplot(gs[1,1])
    acts.set_title('(b) Rounds and swept cells',loc='left',fontweight='bold')
    for method,col,label in [('ddfp_k16',fixed,'K=16'),('ddfp_k16m',scaled,'K=16m')]:
        medA=[]; medQ=[]
        for m in (1,2,4):
            rr=[r for r in data if r['copies']==m and r['method']==('ddfp_k16' if m==1 else method)]
            vals=[r['R_act_total']/1e3 for r in rr]; qs=[r['activation_cell_visits']/1e12 for r in rr]
            dx=-.035 if method=='ddfp_k16' else .035
            for i,(v,q) in enumerate(zip(vals,qs)):
                acts.scatter(m+dx+(i-1)*.025,v,c=col,s=15,alpha=.55,edgecolors='none')
                work.scatter(m+dx+(i-1)*.025,q,c=col,s=15,alpha=.55,edgecolors='none')
            medA.append(st.median(vals)); medQ.append(st.median(qs))
        acts.plot([1,2,4],medA,c=col,lw=1.3,label=label)
        work.plot([1,2,4],medQ,c=col,lw=1.3)
    acts.set_ylabel('Activation sum (thousands)'); work.set_ylabel(r'$Q=\sum_{r,j} A_{rj}P_j$ (trillions)')
    for axis in (acts,work):
        axis.set_xticks([1,2,4],['1','2','4']); axis.set_xlabel('Depth multiplier m'); axis.set_xlim(.7,4.3); axis.grid(axis='y',color='#e3e6e8',lw=.5)
    acts.legend(loc='upper left',frameon=False,fontsize=7)
    work.text(.03,.92,r'$S=K(1+R^*)$',transform=work.transAxes,fontsize=8)
    work.text(.03,.82,'m=4: S=48 vs 192',transform=work.transAxes,fontsize=7)
    work.text(.03,.72,'P: padded cells / slab',transform=work.transAxes,fontsize=7)
    work.text(.03,.62,'A: rounds / solve',transform=work.transAxes,fontsize=7)
    t=fig.add_subplot(gs[0,2]); mem=fig.add_subplot(gs[1,2])
    t.set_title('(c) Time / memory',loc='left',fontweight='bold')
    for method,col,label in [('ddfp_k16',fixed,'K=16 fixed'),('ddfp_k16m',scaled,'K=16m')]:
        vals=[]
        for m in (1,2,4):
            rr=[r for r in a['subject_ratios'] if r['copies']==m and r['method']==('ddfp_k16' if m==1 else method)]
            for i,r in enumerate(rr): t.scatter(m+(i-1)*.035,r['ratio'],s=16,c=col,alpha=.55,edgecolors='none')
            vals.append(math.exp(st.mean(math.log(r['ratio']) for r in rr)))
        t.plot([1,2,4],vals,color=col,lw=1.5,label=label)
        t.text(4.07,vals[-1],f'{vals[-1]:.3f}',va='center',fontsize=8,color=col)
    t.axhline(1,c=neutral,ls='--',lw=.8,label='GPU FP reference')
    t.set_xlim(.7,4.7); t.set_ylim(.88,1.9); t.set_xticks([1,2,4]); t.set_xlabel('Depth multiplier m')
    t.set_ylabel('DD / GPU FP wall time'); t.legend(loc='upper left',frameon=False,fontsize=7); t.grid(axis='y',color='#e3e6e8',lw=.5)
    live=[]; reserve=[]
    for method in ['gpu_k1','ddfp_k16','ddfp_k16m']:
        rr=[r for r in a['rows'] if r['copies']==4 and r['method']==method]
        live.append(st.median(r['pool_live_peak_gib'] for r in rr)); reserve.append(st.median(r['pool_reserved_peak_gib'] for r in rr))
    x=np.arange(3); colors=[neutral,fixed,scaled]
    for i,col in enumerate(colors):
        mem.bar(i,reserve[i],width=.58,facecolor='none',edgecolor=col,lw=1.2,hatch='////')
        mem.bar(i,live[i],width=.34,color=col)
        mem.text(i,reserve[i]+.3,f'{reserve[i]:.2f}',ha='center',fontsize=7,color=col)
        mem.text(i,live[i]-.45,f'{live[i]:.2f}',ha='center',fontsize=7,color='white')
    mem.set_xticks(x,['GPU FP','DD K=16','DD K=64']); mem.set_ylabel('CuPy pool peak (GiB)'); mem.set_ylim(0,14)
    mem.text(.02,.95,'m=4  |  solid: live; outline: reserved',transform=mem.transAxes,fontsize=7)
    mem.grid(axis='y',color='#e3e6e8',lw=.5); mem.set_axisbelow(True)
    fig.subplots_adjust(left=.085,right=.98,bottom=.20,top=.93)
    fig.text(.085,.065,'R8-C: 3 selected subjects; wall time = within-subject median of 3 timing runs.',fontsize=6.5,color=neutral)
    fig.text(.085,.037,'Points: subjects; m=1 policies coincide. Lines: median (b), geometric mean (c).',fontsize=6.5,color=neutral)
    fig.text(.085,.01,'Memory: median of subject peaks. d: halo-inclusive source planes; R*: IBI rounds.',fontsize=6.5,color=neutral)
    for ext in ('svg','png'): fig.savefig(out/f'F1_policy_mechanism.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    writej(out/'figure_metadata.json',{'source_audit_sha256':__import__('r8d_common').sha(ROOT/'results/d0_audit.json'),
        'matplotlib':matplotlib.__version__,'figure_inches':[7.0,5.1],
        'encoding':{'teal':'K=16m policy','amber':'K=16 policy','gray':'GPU reference or neutral structure'},
        'caption':'Core/halo geometry, activation and full-grid scan proxy, and time/memory trade-off in R8-C. At m=4 K64 has more FP calls and activation rounds than K16, but smaller Q and observed pool peaks. Time ratios use subject-level timing medians; memory bars summarize observed metadata peaks.'})
    print(out/'F1_policy_mechanism.png')


if __name__=='__main__': main()
