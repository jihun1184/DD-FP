"""AST copies add observation scopes to the hash-checked IBI control flow.

No disk source changes, no frame.f_locals snapshots, no tensor retention in
records. `with` adds no Python local scope. This is an intrusive profile path;
the baseline path never uses these copies. CPU equivalence is tested.
"""
import ast
import contextlib
import copy
import importlib
import time
from pathlib import Path


def clone(fn, overrides, trace=None):
    tree=ast.parse(Path(fn.__code__.co_filename).read_text(encoding='utf-8'))
    node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fn.__name__))
    def wrap(statements,label):
        ctx=ast.Call(ast.Attribute(ast.Name('__trace',ast.Load()),'span',ast.Load()),[ast.Constant(label)],[])
        return ast.copy_location(ast.With([ast.withitem(ctx,None)],statements),statements[0])
    class Inject(ast.NodeTransformer):
        def visit_For(self,n):
            n=self.generic_visit(n)
            if isinstance(n.target,ast.Tuple) or isinstance(n.target,ast.Name) and n.target.id=='m':
                scope=wrap(n.body,'slab')
                r=ast.Constant(0) if isinstance(n.target,ast.Tuple) else ast.Name('r',ast.Load())
                k=ast.Name('k',ast.Load()) if isinstance(n.target,ast.Tuple) else ast.Subscript(ast.Name('m',ast.Load()),ast.Constant('k'),ast.Load())
                scope.items[0].context_expr=ast.Call(ast.Attribute(ast.Name('__trace',ast.Load()),'slab',ast.Load()),[r,k],[])
                n.body=[scope]
            return n
        def visit_Assign(self,n):
            n=self.generic_visit(n)
            if trace is None: return n
            names=[t.id for t in n.targets if isinstance(t,ast.Name)]
            if 'max_change' in names:
                cb=ast.Expr(ast.Call(ast.Attribute(ast.Name('__trace',ast.Load()),'round_change',ast.Load()),
                    [ast.Name('r',ast.Load()),ast.Name('max_change',ast.Load())],[]))
                return [wrap([n],'convergence'),ast.copy_location(cb,n)]
            if any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id in ('u_new','u_assembled') for t in n.targets):
                return wrap([n],'assembly_interior')
            if names and names[0] in ('u_prev','u_new'):
                return wrap([n],'convergence_copy')
            if names==['seeds']: return wrap([n],'seed_allocation')
            return n
    if trace is not None: node=Inject().visit(node)
    namespace=dict(fn.__globals__); namespace.update(overrides); namespace['__trace']=trace
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(module,fn.__code__.co_filename,'exec'),namespace)
    return namespace[fn.__name__]


class ActivationTrace(contextlib.AbstractContextManager):
    def __init__(self,be,profile=None):
        self.be=be; self.profile=profile; self.solves=[]; self.round_changes=[]; self.restore=[]
        self.current_host_slab=None

    def round_change(self,r,v):
        self.round_changes.append({'round':int(r),'max_change':float(v)})

    def span(self,label):
        return self.profile.span(label) if self.profile else contextlib.nullcontext()

    def slab(self,r,k):
        return self.span(f'slab:r{r}:k{k}')

    def __enter__(self):
        be=self.be; cp=self.profile.cp_proxy if self.profile else be.cp
        old_build,old_fp,old_ibi=be._build,be._fp,be._run_ibi
        def observed(fn,label):
            def run(*args,**kw):
                with self.span(label): return fn(*args,**kw)
            return run
        observed_build=observed(clone(old_build,{'cp':cp}),'build_ispan_gpu')
        def build(*args,**kw):
            # Host slab view only; original vol already lives for the whole run.
            # Never keep endpoint/device arrays or prior solver outputs alive.
            if self.profile: self.current_host_slab=args[0]
            return observed_build(*args,**kw)
        fp=observed(clone(old_fp,{'cp':cp}),'fp_api')
        ns={'cp':cp,'build_ispan_gpu':build,'front_propagation_gpu':fp}
        for name,label in [('_pin_and_seed_expanded','boundary_pin'),('_assemble_boundary','assembly_boundary')]:
            ns[name]=observed(clone(old_ibi.__globals__[name],{'cp':cp}),label)
        be._build=build; be._fp=fp
        be._run_ibi=clone(old_ibi,ns,self)
        self.restore.append(lambda:(setattr(be,'_build',old_build),setattr(be,'_fp',old_fp),setattr(be,'_run_ibi',old_ibi)))
        mod=importlib.import_module('src.ddfp.gpu_seed_aware_worklist'); original=mod.seed_aware_boutry_bucket_gpu
        def solve(*args,**kw):
            args=list(args); args[0]=cp
            seeded=kw.get('boundary_seeds') is not None or len(args)>4 and args[4] is not None
            t=time.perf_counter()
            with self.span('fp_seeded' if seeded else 'fp_unseeded'):
                result=original(*args,**kw)
            meta=result[1]
            contract={'status':'NOT MEASURED on metadata pass'}
            if self.profile:
                from r8c_validate import check_array
                with self.span('fp_contract_verification'):
                    contract=check_array(result[0][1:-1,1:-1,1:-1],self.current_host_slab)
            self.solves.append({'index':len(self.solves),'seeded':seeded,'padded_shape':list(args[1].shape),
                'padded_cells':int(args[1].size),'R_act':int(meta['total_activation_rounds']),
                'scheduler_transitions':int(meta['scheduler_transition_count']),
                'nan_count':int(meta['nan_count']),'observed_wall_s':time.perf_counter()-t,
                'seed_count':meta.get('seed_count'),'seed_raw_bits_preserved':meta.get('seed_raw_bits_preserved'),
                'original_slab_constraints':contract})
            return result
        mod.seed_aware_boutry_bucket_gpu=solve
        self.restore.append(lambda:setattr(mod,'seed_aware_boutry_bucket_gpu',original))
        return self

    def __exit__(self,*exc):
        for fn in reversed(self.restore): fn()
        return False

    def result(self):
        return {'solves':self.solves,'n_solves':len(self.solves),'round_changes':self.round_changes,
                'R_act_total':sum(q['R_act'] for q in self.solves),
                'activation_cell_visits':sum(q['R_act']*q['padded_cells'] for q in self.solves)}
