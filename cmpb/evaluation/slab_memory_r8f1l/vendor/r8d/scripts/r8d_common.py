"""R8-D shared I/O, immutable source fingerprints, and environment admission.

Only standard-library imports are required for D0 and failed preflight.
The byte-identical R8-C helpers are dependencies, never edited in place.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/r8c/scripts'))
import r8c_common as C
from r8c_memory_model import extent, model, slab_shapes, verifier_plan

sha, obj_sha, readj, writej, seal = C.sha, C.obj_sha, C.readj, C.writej, C.seal
KS = {'1': [8,16,32], '2': [8,16,32,64], '4': [8,16,32,64,128]}


def config(path):
    c = C.load_config(path)
    if c.get('k_candidates') != KS:
        raise ValueError('registered BraTS candidate sets differ')
    if c.get('stage_timeout_s', 0) <= 0:
        raise ValueError('positive stage_timeout_s required')
    return c


def output(path):
    p = Path(path).resolve()
    if p == ROOT/'results' or not p.is_relative_to((ROOT/'results').resolve()):
        raise ValueError('output must be a named child of R8-D/results')
    return p


def hashes(c):
    h = C.source_hashes(c)
    for folder in ('scripts', 'vendor'):
        for p in sorted((ROOT/folder).rglob('*.py')):
            h[str(p)] = sha(p)
    # Refuse vendor drift relative to the D0 recovered bytes.
    for rel, expected in readj(ROOT/'config/vendor_hashes.json').items():
        if sha(ROOT/rel) != expected:
            raise ValueError('vendor fingerprint changed: '+rel)
    return h


def gate(c, require_data=True):
    result = C.gate(c, require_data=require_data)
    try:
        result['sources'] = hashes(c)
    except Exception as e:
        result['reasons'].append(str(e))
    result['ok'] = not result['reasons']
    return result


def load_r7(c):
    sys.path.insert(0, str(Path(c['r7_runner'])/'r7'))
    import r7_common
    return r7_common


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    return p


def arr_sha(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    return hashlib.sha256(memoryview(a).cast('B')).hexdigest()
