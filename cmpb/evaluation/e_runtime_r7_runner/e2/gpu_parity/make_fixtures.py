"""Builds the FIXED parity fixture set (counterexample + 49 = 50 total) from vendor/cpu only. Run once (CPU, numpy); fixtures.json is then frozen.
Usage: python make_fixtures.py"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
V = Path(__file__).resolve().parent / "vendor"; sys.path.insert(0, str(V / "cpu"))
from cpu_fp import build_ispan_cpu; from seedref import seed_aware_cpu_reference as seeded; import ibi_cpu as I

rng = np.random.default_rng(20261013); fx = []
cx = json.loads((V / "min_counterexample.json").read_text())
fx.append(dict(id="F00_cx_3x3x5", kind="ibi", K=cx["K"], delta=cx["delta"], vol=cx["vol"]))
def rv(shp, a):
    v = rng.integers(0, a, shp).astype(np.int64); return (v * (255 // max(1, int(v.max())))).astype(np.uint8)
# F01-F20: IBI end-to-end, mixed alphabets/shapes/K/delta
for i in range(20):
    shp = (int(rng.integers(3, 5)), int(rng.integers(3, 5)), int(rng.integers(5, 10))); a = int(rng.choice([2, 3, 4, 256]))
    fx.append(dict(id=f"F{i+1:02d}_ibi_a{a}", kind="ibi", K=int(rng.integers(2, 4)), delta=int(rng.integers(1, 3)), vol=rv(shp, a).tolist()))
# F21-F40: single tile solve with random interior pins (singleton intervals + seed)
for i in range(20):
    shp = (3, 3, int(rng.integers(4, 7))); a = int(rng.choice([3, 4, 256])); vol = rv(shp, a); lo, hi, l = build_ispan_cpu(vol)
    inner = [(x, y, z) for x in range(1, lo.shape[0]-1) for y in range(1, lo.shape[1]-1) for z in range(1, lo.shape[2]-1)]
    sel = rng.choice(len(inner), size=int(rng.integers(2, 9)), replace=False)
    pins = [[*inner[j], int(rng.integers(0, 256))] for j in sel]
    fx.append(dict(id=f"F{21+i:02d}_tile_pins_a{a}", kind="tile", vol=vol.tolist(), pins=pins))
# F41-F49: tile solves in which the CPU reference records >=1 equidistant next-level TIE EVENT (alphabet 8; ~1.4% of tiles). Alternating unseeded / pinned.
n = 0
while n < 9:
    shp = (int(rng.integers(3, 5)), 3, int(rng.integers(4, 6))); vol = rv(shp, 8); lo, hi, l = build_ispan_cpu(vol); pins = []
    if n % 2:
        inner = [(x, y, z) for x in range(1, lo.shape[0]-1) for y in range(1, lo.shape[1]-1) for z in range(1, lo.shape[2]-1)]
        pins = [[*inner[j], int(rng.integers(0, 256))] for j in rng.choice(len(inner), size=2, replace=False)]
        for x, y, z, val in pins: lo[x, y, z] = hi[x, y, z] = val
    seeds = None
    if pins:
        seeds = np.full(lo.shape, np.nan, np.float32)
        for x, y, z, val in pins: seeds[x, y, z] = val
    ev = seeded(lo, hi, l, seeds, "lower")[1]["equidistant_next_level_tie_events"]
    if ev > 0:
        fx.append(dict(id=f"F{41+n:02d}_tile_tie_events{ev}", kind="tile", vol=vol.tolist(), pins=pins)); n += 1
assert len(fx) == 50
body = json.dumps(fx, separators=(",", ":"))
Path(__file__).with_name("fixtures.json").write_text(json.dumps(dict(n=len(fx), seed=20261013, sha256_of_list=hashlib.sha256(body.encode()).hexdigest(), fixtures=fx), indent=0))
print("fixtures:", len(fx), hashlib.sha256(body.encode()).hexdigest())
