"""R8-C V1 static tensor model; no CUDA imports and no fabricated observations.

Run: python -B scripts/r8c_memory_model.py --shape 240 240 155 --K 16 --json results/model.json
Only float32 production is supported. --reserve-factor is an EXPLICIT scenario,
not a validated predictor. A None reserve-factor returns NOT IDENTIFIED.
"""
import argparse
import json
import math
import sys

I32 = 2**31 - 1
MIB = 2**20


def shape3(shape):
    if len(shape) != 3 or any(type(x) is not int or x < 1 for x in shape):
        raise ValueError('shape must contain exactly three positive integers')
    return tuple(shape)


def extent(shape, pad=False):
    return tuple(2*x + (1 if pad else -1) for x in shape3(shape))


def verifier_plan(shape):
    w, h, d = shape3(shape)
    q = {'W': w, 'H': h, 'D': d, 'index_extent': w*h*d,
         'xy': (w-1)*(h-1)*d, 'xz': (w-1)*h*(d-1), 'yz': w*(h-1)*(d-1),
         'cube': (w-1)*(h-1)*(d-1)}
    q['faces'] = q['xy'] + q['xz'] + q['yz']
    q['total_blocks'] = q['faces'] + q['cube']
    q['max_thread_index'] = ((q['total_blocks'] + 255)//256)*256 - 1
    # Degenerate shapes are excluded: the production launch may have grid zero.
    ok = min(w, h, d) >= 2 and max(q.values()) <= I32
    return {'ok': ok, 'verify_status': 'SAFE_INT32' if ok else 'REFUSED_INT32', 'quantities': q}


def slab_shapes(shape, K, delta):
    w, h, d = shape3(shape)
    if type(K) is not int or not 1 <= K <= d or type(delta) is not int or delta < 0:
        raise ValueError('require 1 <= K <= D and integer delta >= 0')
    base, rem = divmod(d, K); z = 0; out = []
    for k in range(K):
        size = base + (k < rem)
        out.append((w, h, min(d, z+size+delta)-max(0, z-delta))); z += size
    return out


def build(shape):
    w, h, d = shape3(shape)
    n0, ne, npad = math.prod(shape), math.prod(extent(shape)), math.prod(extent(shape, True))
    # At second pad: face lo/hi locals (last orientation) remain alive,
    # as do 8-corner stack c and s and unpadded endpoint arrays.
    face = w*(h-1)*(d-1); corners = (w-1)*(h-1)*(d-1)
    pad_stage = 4*n0 + 8*ne + 8*face + 32*corners + 8*npad
    # A conservative visible-expression envelope for min/max RHS nesting,
    # c reductions and pad (internal CuPy reduction workspace not included).
    expression = 4*n0 + 8*ne + max(24*face, 32*corners+8*face+4*corners)
    return {'pad_live_bytes': pad_stage, 'visible_live_envelope_bytes': max(pad_stage, expression),
            'last_face_locals_bytes': 8*face, 'corner_stack_bytes': 32*corners}


def model(shape, K=1, delta=1, rounds=2, dtype='float32', reserve_factor=None, reserve_fixed_mib=0.0, context_mib=0.0):
    shape = shape3(shape)
    if dtype != 'float32': raise ValueError('frozen kernels require float32 (4 bytes); dtype conversion changes algorithm contract')
    if type(rounds) is not int or rounds < 0: raise ValueError('rounds must be a nonnegative integer')
    if any(not math.isfinite(x) or x < 0 for x in (reserve_fixed_mib, context_mib)):
        raise ValueError('overheads must be finite and nonnegative')
    if reserve_factor is not None and (not math.isfinite(reserve_factor) or reserve_factor < 1):
        raise ValueError('reserve-factor must be finite and >= 1')
    slabs = slab_shapes(shape, K, delta)
    ne, npad = math.prod(extent(shape)), math.prod(extent(shape, True))
    ledger = []; peaks = {}
    if K == 1:
        peaks = {'build': build(shape)['visible_live_envelope_bytes'], 'fp': 28*npad+1084, 'reupload': 12*npad}
        # host endpoints (8Np) + largest expression temporaries (24Np).
        host = 32*npad + 4*ne
        global_b, workspace = 0, max(peaks.values())
        ledger = [{'name': n, 'dtype': 'float32' if n in ('lo', 'hi', 'u') else 'int32',
                   'shape': extent(shape, True), 'bytes': 4*npad} for n in ('lo', 'hi', 'u', 'discovered', 'processed', 'scheduled', 'frontier')]
    else:
        prev = None; r0 = 0; r1 = 0
        for s in slabs:
            ns = math.prod(extent(s, True)); pn = math.prod(extent(prev, True)) if prev else 0
            bs = build(s)['visible_live_envelope_bytes']
            r0 = max(r0, 4*ne + max(bs+12*pn, 28*ns+4*pn+1084))
            # On round>=1 entry, the previous round's last slab is still live.
            prev = s
        prev = slabs[-1]
        for s in slabs:
            ns = math.prod(extent(s, True)); pn = math.prod(extent(prev, True))
            r1 = max(r1, 12*ne + max(build(s)['visible_live_envelope_bytes']+16*pn,
                                    12*ns+8*pn, 36*ns+4*pn+1084))
            prev = s
        lastn = math.prod(extent(slabs[-1], True))
        peaks = {'round0': r0}
        if rounds:
            peaks.update(round_slabs=r1, convergence=20*ne+16*lastn)
        host = 64*max(math.prod(extent(s, True)) for s in slabs) + 8*ne
        # 64 B/slab-cell is a declared conservative HOST scenario, not exact RSS.
        global_b = (20 if rounds else 4)*ne
        workspace = max(peaks.values())-global_b
        ledger = [{'name': 'u_assembled/u_prev/u_new', 'dtype': 'float32', 'shape': extent(shape), 'bytes_each': 4*ne,
                   'unique_count_during_slab': 3 if rounds else 1},
                  {'name': 'subtract/abs', 'dtype': 'float32', 'shape': extent(shape), 'bytes_each': 4*ne, 'count_at_check': 2 if rounds else 0}]
    live = max(peaks.values())
    reserve = None if reserve_factor is None else live*reserve_factor + reserve_fixed_mib*MIB
    fp_safe = all(math.prod(extent(s, True)) <= I32 for s in (slabs if K > 1 else [shape]))
    return {'model': 'R8C_V1_VISIBLE_TENSORS', 'kind': 'STATIC_PREDICTION', 'shape': shape, 'K': K, 'delta': delta, 'rounds': rounds,
            'dtype': dtype, 'expanded_cells': ne, 'padded_cells': npad, 'slab_shapes': slabs,
            'live_visible_peak_bytes': live, 'live_visible_peak_mib': live/MIB, 'stage_bytes': peaks,
            'global_check_state_bytes': global_b, 'slab_or_fp_workspace_bytes': workspace,
            'global_persistent_peak_bytes': (12 if rounds else 4)*ne if K>1 else 0,
            'global_convergence_temporary_bytes': 8*ne if K>1 and rounds else 0,
            'host_scenario_bytes': host, 'host_scope': 'array envelope only; excludes interpreter, retained outputs, I/O and allocator RSS',
            'pool_reserved_prediction_bytes': reserve, 'reserved_status': 'NOT_IDENTIFIED' if reserve is None else 'ASSUMPTION_SCENARIO_NOT_VALIDATED',
            'assumptions': {'reserve_factor': reserve_factor, 'reserve_fixed_mib': reserve_fixed_mib, 'context_mib': context_mib,
                            'excludes': ['internal reduction workspace', 'allocator rounding/fragmentation', 'non-pool allocations']},
            'device_prediction_mib': None if reserve is None else reserve/MIB+context_mib,
            'array_ledger': ledger, 'fp_int32_ok': fp_safe, 'global_verifier': verifier_plan(extent(shape)),
            'capacity_limit': 'CONDITIONAL: not inferable from live tensors without allocator/device/host budgets'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shape', nargs=3, type=int, default=[240, 240, 155]); ap.add_argument('--K', type=int, default=1)
    ap.add_argument('--delta', type=int, default=1); ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--dtype', default='float32'); ap.add_argument('--reserve-factor', type=float)
    ap.add_argument('--reserve-fixed-mib', type=float, default=0); ap.add_argument('--context-mib', type=float, default=0); ap.add_argument('--json')
    a = ap.parse_args()
    try: out = model(a.shape, a.K, a.delta, a.rounds, a.dtype, a.reserve_factor, a.reserve_fixed_mib, a.context_mib)
    except (ValueError, TypeError) as e: ap.error(str(e))
    text = json.dumps(out, indent=2)
    if a.json:
        from pathlib import Path
        Path(a.json).write_text(text+'\n', encoding='utf-8')
    print(text)


if __name__ == '__main__': main()
