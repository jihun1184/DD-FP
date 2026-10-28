"""CPU unit/negative controls. Synthetic fixtures below are NEVER research data."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
import r8c_common as C
import r8c_memory_model as M
import r8c_validate as V
import r8c_scaling_runner as S
import r8c_analyze as A


def fixture():
    """Deliberately marked fake even when testing the positive record schema."""
    return {'schema':'R8C-1','status':'COMPLETED','backend':'cupy','cuda_executed':True,'purpose':'metadata',
            'subject':'SYNTHETIC_UNIT_TEST','copies':1,'rep':0,'method':'ddfp_k16','K':16,'vol_shape':[9,9,48],
            'out_shape':[17,17,95],'input_sha256':'a'*64,'vol_sha256':'b'*64,'config_sha256':'c'*64,'out_sha256':'d'*64,
            'sources_unchanged':True,'output_npy':'output.npy','output_file_sha256':'e'*64,
            'array_check':{'ok':True},'verification':{'verify_status':'DONE','global_dwc_ok':True,'global_violations':0,'nan_skipped':0,'seam_violations':0},
            'R_star':2,'max_rounds':8,'metadata':{'n_solves':48,'R_act_total':960,'round_changes':[{'round':1,'max_change':1.0},{'round':2,'max_change':0.0}]},'foreign_gpu_procs':[],'regression':{'ok':True},'t_algorithm_s':1.,'wall_s':1.,
            'memory':{'sampler_errors':[],'foreign_observed':[],'n_samples':10,'pool_used_peak_bytes':1,'pool_total_peak_bytes':2,'host_hwm_bytes':3,'baseline_device_used_bytes':0,
                      'sampled_host_rss_peak_bytes':3,'sampled_device_delta_peak_bytes':2,'hwm_reset':True,'host_hwm_scope':'fixture','pool_peak_scope':'fixture'}}


class ModelTests(unittest.TestCase):
    def test_geometry(self):
        self.assertEqual(math.prod(M.extent((240,240,155))),70_897_269)
        self.assertEqual(math.prod(M.extent((240,240,155),True)),71_953_271)
    def test_float32_contract(self):
        for sh in [(0,2,3),(True,2,3),(2.5,3,4),(2,3)]:
            with self.assertRaises((ValueError,TypeError)): M.model(sh)
        for kwargs in [{'K':0},{'K':156},{'delta':-1},{'rounds':-1},{'dtype':'float16'},{'reserve_factor':.9},{'reserve_factor':float('nan')}]:
            with self.assertRaises(ValueError): M.model((240,240,155),**kwargs)
    def test_model_live_vs_pool(self):
        r=M.model((240,240,155)); self.assertIsNone(r['pool_reserved_prediction_bytes'])
        q=M.model((240,240,155),reserve_factor=1.8,reserve_fixed_mib=256)
        self.assertAlmostEqual(q['pool_reserved_prediction_bytes'],1.8*r['live_visible_peak_bytes']+256*2**20)
        self.assertEqual(r['stage_bytes']['fp'],28*71_953_271+1084)
    def test_build_missing_locals(self):
        sh=(240,240,155); w,h,d=sh
        old=4*math.prod(sh)+8*math.prod(M.extent(sh))+32*(w-1)*(h-1)*(d-1)+8*math.prod(M.extent(sh,True))
        self.assertEqual(M.build(sh)['pad_live_bytes']-old,8*w*(h-1)*(d-1))
    def test_slabs(self):
        ss=M.slab_shapes((240,240,155),16,1)
        self.assertEqual(len(ss),16); self.assertEqual(sorted(set(s[2] for s in ss)),[10,11,12])
        self.assertEqual(M.model((240,240,620),64)['K'],64)
    def test_int32_boundary_and_degenerate(self):
        for m in (1,2,4,6,7): self.assertTrue(M.verifier_plan((479,479,310*m-1))['ok'])
        r=M.verifier_plan((479,479,2479)); self.assertFalse(r['ok']); self.assertEqual(r['quantities']['total_blocks'],2_267_330_860)
        self.assertFalse(M.verifier_plan((1,1,1))['ok'])
    def test_r8b_differential_valid_domain(self):
        p=ROOT/'frozen/NEUROCOMPUTING_V1_INTEGRITY_RECOVERY/20261013/R8B_SCALING_EXECUTION/r8_guard.py'
        sp=importlib.util.spec_from_file_location('guard_for_test',p); mod=importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
        rng=random.Random(42)
        for _ in range(2000):
            sh=tuple(rng.randint(2,3000) for _ in range(3)); self.assertEqual(M.verifier_plan(sh)['ok'],mod.verify_plan(sh)['ok'])


class RecordTests(unittest.TestCase):
    def test_schema_positive_fixture(self): self.assertTrue(V.validate_record(fixture())['ok'])
    def test_negative_controls(self):
        for key,val in [('status','ERROR'),('backend','stub'),('cuda_executed',False),('out_sha256','fake'),('foreign_gpu_procs',None),
                        ('R_star',8),('t_algorithm_s',float('nan')),('out_shape',[1,1,1]),('regression',{}),('array_check',{})]:
            r=fixture(); r[key]=val; self.assertFalse(V.validate_record(r)['ok'],(key,val))
        for key,val in [('global_dwc_ok',False),('global_violations',1),('nan_skipped',1),('seam_violations',1),('verify_status','REFUSED_INT32')]:
            r=fixture(); r['verification'][key]=val; self.assertFalse(V.validate_record(r)['ok'],key)
    def test_rstar3_is_valid_but_cap_is_not(self):
        r=fixture(); r['R_star']=3; r['metadata']['n_solves']=64
        r['metadata']['round_changes']=[{'round':1,'max_change':2.0},{'round':2,'max_change':1.0},{'round':3,'max_change':0.0}]
        self.assertTrue(V.validate_record(r)['ok'])
        r['metadata']['n_solves']=48; self.assertFalse(V.validate_record(r)['ok'])
    def test_unsafe_size_no_false_pass(self):
        r=fixture(); r['vol_shape']=[240,240,1240]; r['out_shape']=[479,479,2479]
        self.assertFalse(V.validate_record(r)['ok'])
        r['verification']={'verify_status':'REFUSED_INT32'}; self.assertFalse(V.validate_record(r)['ok'])
    def test_memory_monitor_errors_rejected(self):
        for key,val in [('sampler_errors',['unavailable']),('foreign_observed',[{'pid':999}]),('n_samples',0)]:
            r=fixture(); r['memory'][key]=val; self.assertFalse(V.validate_record(r)['ok'])
    def test_malformed_record_and_metrics(self):
        for r in (None,[],{'memory':[]},{}): self.assertFalse(V.validate_record(r)['ok'])
        for val in ('bad',None,float('nan'),0):
            r=fixture(); r['memory']['pool_used_peak_bytes']=val; self.assertFalse(V.validate_record(r)['ok'])
    def test_corrupt_json_is_not_pass(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as td:
            p=Path(td)/'workers/bad'; p.mkdir(parents=True); (p/'record.json').write_text('{broken')
            self.assertEqual(V.validate_run(td)['status'],'FAIL_OR_INCOMPLETE')
    def test_missing_smoke_and_mixed_config_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as td:
            d=Path(td); cfg={'test_fixture':True}; cfgsha=C.obj_sha(cfg)
            C.writej(d/'run_config.json',{'subjects':['a','b','c'],'config':cfg,'config_sha256':cfgsha,'source_hashes':{}})
            for j in S.plan_jobs({'copies':[1,2,4],'reps':3,'order_seed':1},['a','b','c']):
                r=fixture(); r.update({k:j[k] for k in ('subject','copies','method','purpose','rep')})
                r['K']=1 if j['method']=='gpu_k1' else 16 if j['method']=='ddfp_k16' else 16*j['copies']
                C.writej(d/'workers'/j['id']/'record.json',r)
            q=V.validate_run(d); self.assertFalse(q['smoke_gate_ok']); self.assertTrue(q['provenance_errors'])
            self.assertEqual(q['status'],'FAIL_OR_INCOMPLETE')
    def test_complete_synthetic_schema_and_smoke_removal(self):
        """100 synthetic records confined to auto-deleted temp; tests schema only."""
        import numpy as np
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as td:
            d=Path(td); cfg={'TEST_FIXTURE_ONLY':True}; cfgsha=C.obj_sha(cfg); sources={'TEST_FIXTURE_ONLY':'f'*64}
            inventory={s:{'sha256':'a'*64,'vol_shape':[3,3,48]} for s in ('a','b','c')}
            C.writej(d/'run_config.json',{'subjects':['a','b','c'],'config':cfg,'config_sha256':cfgsha,'source_hashes':sources})
            jobs=S.plan_jobs({'copies':[1,2,4],'reps':3,'order_seed':1},['a','b','c'])
            jobs += [{'id':f'SMOKE_{me}_{i}','subject':'SMOKE','copies':1,'method':me,'purpose':'smoke','rep':i} for me in ('gpu_k1','ddfp_k16') for i in (0,1)]
            sg=[]
            for j in jobs:
                folder=d/'workers'/j['id']; folder.mkdir(parents=True)
                r=fixture(); r.update({k:j[k] for k in ('subject','copies','method','purpose','rep')}); r['config_sha256']=cfgsha
                r['K']=1 if j['method']=='gpu_k1' else 16 if j['method']=='ddfp_k16' else 16*j['copies']
                r['vol_shape']=[3,3,48*j['copies']]; u=np.zeros(M.extent(tuple(r['vol_shape'])),dtype=np.float32); r['out_shape']=list(u.shape)
                r['out_sha256']=hashlib.sha256(memoryview(u).cast('B')).hexdigest(); r['metadata']['n_solves']=1 if r['K']==1 else r['K']*3
                if j['copies']>1: r['regression']={'status':'NOT_AVAILABLE'}
                if j['purpose']=='metadata':
                    np.save(folder/'output.npy',u,allow_pickle=False); r['output_file_sha256']=C.sha(folder/'output.npy')
                else: r.pop('output_npy'); r.pop('output_file_sha256')
                if j['purpose']=='smoke':
                    r['cpu_independent_dwc_violations']=0; r['verifier_negative_control']={'detected':True}
                    sg.append([r['method'],r['rep'],r['out_sha256']])
                C.writej(folder/'environment.json',{'ok':True,'config_sha256':cfgsha,'sources':sources,'inventory':inventory})
                r['environment_sha256']=C.sha(folder/'environment.json'); C.writej(folder/'record.json',r)
            C.writej(d/'smoke_gate.json',{'status':'PASS','records':sg})
            good=V.validate_run(d); self.assertEqual(good['status'],'PASS',str(good)[:500])
            self.assertEqual(good['n_primary_records'],96); self.assertEqual(good['n_records'],100)
            (d/'smoke_gate.json').unlink(); self.assertFalse(V.validate_run(d)['smoke_gate_ok'])
    def test_empty_run_and_fake_smoke_no_scaling(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as td:
            self.assertEqual(V.validate_run(td)['status'],'INCONCLUSIVE')
            self.assertEqual(A.analyze(td)['scientific_assessment'],'INCONCLUSIVE')
    def test_missing_saved_file(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as td:
            r=fixture(); r['output_npy']='missing.npy'; self.assertFalse(V.validate_record(r,td)['ok'])


class ArrayTests(unittest.TestCase):
    def test_full_interval_check_and_mutations(self):
        import numpy as np
        vol=np.random.default_rng(9).integers(0,256,(5,6,7),dtype=np.uint8)
        lo=np.zeros(M.extent(vol.shape),np.float32)
        # Independently fill each cell from its geometric incident corners.
        for i in range(lo.shape[0]):
            for j in range(lo.shape[1]):
                for k in range(lo.shape[2]):
                    lo[i,j,k]=vol[i//2:(i+1)//2+1,j//2:(j+1)//2+1,k//2:(k+1)//2+1].min()
        self.assertTrue(V.check_array(lo,vol)['ok'])
        b=lo.copy(); b[1,1,1]=300; q=V.check_array(b,vol); self.assertEqual(q['original_interval_violations'],1)
        b=lo.copy(); b[0,0,0]+=1; self.assertEqual(V.check_array(b,vol)['original_voxel_differences'],1)
        b=lo.copy(); b[-1,-1,-1]=np.nan; self.assertFalse(V.check_array(b,vol)['ok'])
        self.assertFalse(V.check_array(lo.astype(np.float64),vol)['ok'])


class ControllerTests(unittest.TestCase):
    def test_fixed_plan_alias_and_reps(self):
        c={'order_seed':20261016,'copies':[1,2,4],'reps':3}; p=S.plan_jobs(c,['a','b','c'])
        self.assertEqual(len(p),96); self.assertEqual(p,S.plan_jobs(c,['a','b','c']))
        self.assertFalse(any(j['copies']==1 and j['method']=='ddfp_k16m' for j in p))
    def test_output_protection(self):
        for p in [ROOT,ROOT/'results',ROOT/'frozen/x','/tmp/outside-r8c']:
            with self.assertRaises(ValueError): C.output_path(p)
        self.assertTrue(C.output_path(ROOT/'results/test').is_relative_to(ROOT/'results'))
    def test_first_admission_cannot_bypass(self):
        c={'delta':1,'reserve_factor':1.8,'reserve_fixed_mib':256,'vram_fraction':.75,'headroom_mib':3072,'host_fraction':.6}
        with patch.object(C,'smi',return_value={'total_mib':2,'used_mib':0,'foreign':[]}),patch.object(C,'host_memory',return_value={'MemAvailable':2**40}):
            self.assertFalse(C.admit(c,(240,240,155),1)['ok'])
    def test_foreign_and_host_refusal(self):
        c={'delta':1,'reserve_factor':1.8,'reserve_fixed_mib':256,'vram_fraction':.75,'headroom_mib':3072,'host_fraction':.6}
        with patch.object(C,'smi',return_value={'total_mib':24000,'used_mib':400,'foreign':[123]}),patch.object(C,'host_memory',return_value={'MemAvailable':100}):
            r=C.admit(c,(240,240,155),1); self.assertFalse(r['ok']); self.assertIn('foreign GPU process present',r['reasons'])
    def test_missing_config(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as td:
            p=Path(td)/'bad.json'; p.write_text('{}')
            with self.assertRaises(ValueError): C.load_config(p)


if __name__=='__main__': unittest.main()
