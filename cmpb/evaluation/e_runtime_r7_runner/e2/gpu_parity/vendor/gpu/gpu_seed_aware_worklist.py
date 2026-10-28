#!/usr/bin/env python3
"""P2-validated seed-aware GPU-native worklist production adapter.

The ``None`` and all-NaN boundary-seed paths delegate to the frozen canonical
native-v1.1 implementation. Only a genuinely seeded call uses the kernels
defined here. P3 changes only the canonical module's repository-relative path;
the validated scheduling and ghost-z admission logic are preserved.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_NATIVE_V1_1_SHA256 = (
    "853bf33a6bbd9c91f5304a018f0a51ff53e8418360fe1477208cf81df17163ac"
)
IMPLEMENTATION_VERSION = "P2-S2c-seed-aware-gpu-prototype-v1+production-right-ghost-z-v1"
N_LEVELS = 256
DEFAULT_BLOCK = 256


_SEED_INIT_KERNEL_CODE = r'''
extern "C" __global__ void initialize_seeds(
    const float* boundary_seeds,
    float* u,
    int* discovered,
    int* scheduled_level,
    int* is_seed,
    int* pending_count,
    unsigned int* pending_bits,
    int field_v_min,
    int field_v_max,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float raw = boundary_seeds[idx];
    if (isnan(raw)) return;

    // Ownership, raw output fixation, and propagation scheduling are distinct.
    u[idx] = raw;
    discovered[idx] = 1;
    is_seed[idx] = 1;
    int level = __float2int_rn(raw);
    if (level < field_v_min) level = field_v_min;
    if (level > field_v_max) level = field_v_max;
    scheduled_level[idx] = level;
    atomicAdd(&pending_count[level], 1);
    atomicOr(&pending_bits[level >> 5], 1u << (level & 31));
}
'''


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_canonical(canonical_path: Path | None = None):
    path = canonical_path or (
        Path(__file__).resolve().parent
        / "gpu_native_worklist.py"
    )
    path = path.resolve()
    actual = sha_file(path)
    if actual != CANONICAL_NATIVE_V1_1_SHA256:
        raise RuntimeError(
            "canonical native-v1.1 hash mismatch: "
            f"expected {CANONICAL_NATIVE_V1_1_SHA256}, got {actual}"
        )
    return import_by_path("_p2s2c_canonical_native_v1_1", path)


def _extract_cuda_function(code: str, name: str) -> str:
    marker = f"__global__ void {name}("
    start = code.index(marker)
    brace = code.index("{", start)
    depth = 0
    for pos in range(brace, len(code)):
        if code[pos] == "{":
            depth += 1
        elif code[pos] == "}":
            depth -= 1
            if depth == 0:
                return code[start : pos + 1]
    raise ValueError(f"unterminated CUDA function: {name}")


def _seeded_traversal_kernel_code(canonical_code: str) -> str:
    """Derive seeded traversal code with one activation semantic change.

    Expansion and next-level selection remain byte-identical to canonical v1.1.
    Unique anchors make lineage drift fail closed.
    """
    signature_anchor = """    const int* scheduled_level,
    int* frontier,"""
    signature_replacement = """    const int* scheduled_level,
    const int* is_seed,
    int* frontier,"""
    write_anchor = "        u[idx] = (float)current_level;"
    write_replacement = """        if (!is_seed[idx]) {
            u[idx] = (float)current_level;
        }"""
    if canonical_code.count(signature_anchor) != 1:
        raise RuntimeError("canonical activation signature anchor is not unique")
    if canonical_code.count(write_anchor) != 1:
        raise RuntimeError("canonical activation write anchor is not unique")
    seeded = canonical_code.replace(signature_anchor, signature_replacement)
    seeded = seeded.replace(write_anchor, write_replacement)
    for name in ("expand_frontier", "select_next_level"):
        if _extract_cuda_function(seeded, name) != _extract_cuda_function(canonical_code, name):
            raise RuntimeError(f"forbidden change to canonical kernel: {name}")
    return seeded


def _boundary_seed_host(cp, boundary_seeds, expected_shape: tuple[int, ...]):
    seeds_dev = cp.ascontiguousarray(cp.asarray(boundary_seeds, dtype=cp.float32))
    seeds = np.ascontiguousarray(cp.asnumpy(seeds_dev).reshape(seeds_dev.shape))
    if seeds.shape != expected_shape or seeds.ndim != 3:
        raise ValueError("boundary_seeds must have the same padded 3-D shape as U_lo/U_hi")
    mask = ~np.isnan(seeds)
    return seeds_dev, seeds, mask


def _outer_padding_mask(shape: tuple[int, int, int]) -> np.ndarray:
    outer = np.zeros(shape, dtype=bool)
    outer[0, :, :] = outer[-1, :, :] = True
    outer[:, 0, :] = outer[:, -1, :] = True
    outer[:, :, 0] = outer[:, :, -1] = True
    return outer


def _validate_production_outer_seed_extension(seed_mask: np.ndarray) -> None:
    """Fail closed outside the exact production right ghost-z extension.

    The pre-existing P2-S2 domain permits arbitrary *interior* seed masks.
    This extension broadens only the outer-padding part of that domain:

    * x/y outer faces remain forbidden;
    * the left z outer face (z=0) remains forbidden;
    * if any outer seed exists, it must be the complete interior-x/y footprint
      on the right ghost z face (z=-1);
    * that ghost plane must be accompanied by the complete adjacent interior
      z plane (z=-2), as produced by a nonempty right-overlap slab.

    Source/Jacobi provenance is intentionally checked by the capture layer,
    because it cannot be inferred from the three solver input arrays alone.
    """
    outer_seed = seed_mask & _outer_padding_mask(seed_mask.shape)
    if not np.any(outer_seed):
        return

    if seed_mask.ndim != 3 or min(seed_mask.shape) < 3:
        raise ValueError("production ghost-z extension requires a padded 3-D grid with interior cells")

    allowed_outer = np.zeros(seed_mask.shape, dtype=bool)
    allowed_outer[1:-1, 1:-1, -1] = True
    if np.any(outer_seed & ~allowed_outer):
        raise ValueError(
            "outer seeds are permitted only on the production right ghost-z face; "
            "x/y outer faces and left z padding remain forbidden"
        )

    if not np.all(seed_mask[1:-1, 1:-1, -1]):
        raise ValueError("production right ghost-z seed must cover the full interior x-y footprint")
    if not np.all(seed_mask[1:-1, 1:-1, -2]):
        raise ValueError("production right ghost-z seed requires the adjacent right-overlap interior plane")


def _validate_seeded_host(
    seeds: np.ndarray,
    seed_mask: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[int, int, np.ndarray]:
    values = seeds[seed_mask]
    if values.size == 0:
        raise ValueError("seeded validation requires at least one non-NaN seed")
    if not np.all(np.isfinite(values)):
        raise ValueError("all non-NaN boundary seeds must be finite")
    frac = float(np.max(np.abs(values.astype(np.float64) - np.rint(values.astype(np.float64)))))
    if frac > 1e-7:
        raise ValueError(
            "P2-S2 paper path requires integer-valued boundary seeds; "
            f"max fractional deviation={frac}"
        )
    if float(values.min()) < 0 or float(values.max()) > 255:
        raise ValueError("P2-S2 paper path requires boundary seeds in [0,255]")
    _validate_production_outer_seed_extension(seed_mask)
    if not np.array_equal(values, lo[seed_mask]) or not np.array_equal(values, hi[seed_mask]):
        raise ValueError("every paper-path seed must equal both pinned singleton endpoints")

    # Use actual endpoint-derived bounds.  This is observationally equivalent
    # to [0,255] in the frozen paper domain but preserves the legacy expression.
    field_v_min = int(np.rint(float(lo.min())))
    field_v_max = int(np.rint(float(hi.max())))
    levels = np.full(lo.shape, -1, dtype=np.int32)
    levels[seed_mask] = np.clip(
        np.rint(values), field_v_min, field_v_max
    ).astype(np.int32)
    return field_v_min, field_v_max, levels


def _kernel_bundle(cp, canonical_code: str):
    traversal = cp.RawModule(
        code=_seeded_traversal_kernel_code(canonical_code),
        options=("-std=c++17",),
    )
    seed_module = cp.RawModule(code=_SEED_INIT_KERNEL_CODE, options=("-std=c++17",))
    return {
        "seed_init": seed_module.get_function("initialize_seeds"),
        "activate": traversal.get_function("activate_level"),
        "expand": traversal.get_function("expand_frontier"),
        "select": traversal.get_function("select_next_level"),
    }


def _scalar_int(cp, value) -> int:
    return int(cp.asnumpy(value).reshape(-1)[0])


def seed_aware_boutry_bucket_gpu(
    cp,
    U_lo_pad,
    U_hi_pad,
    l_inf: float,
    boundary_seeds=None,
    tie_policy: str = "lower",
    block_size: int = DEFAULT_BLOCK,
    *,
    canonical_module=None,
    canonical_path: Path | None = None,
):
    """Run the P2-S2c prototype or delegate the exact unseeded path."""
    canonical = canonical_module or load_canonical(canonical_path)
    if boundary_seeds is None:
        return canonical.boutry_bucket_gpu(
            cp, U_lo_pad, U_hi_pad, l_inf,
            tie_policy=tie_policy, block_size=block_size,
        )

    expected_shape = tuple(int(v) for v in U_lo_pad.shape)
    seeds_dev, seeds_host, seed_mask = _boundary_seed_host(cp, boundary_seeds, expected_shape)
    if not np.any(seed_mask):
        return canonical.boutry_bucket_gpu(
            cp, U_lo_pad, U_hi_pad, l_inf,
            tie_policy=tie_policy, block_size=block_size,
        )
    if tie_policy not in ("lower", "higher"):
        raise ValueError("tie_policy must be 'lower' or 'higher'")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")

    lo_dev = cp.ascontiguousarray(cp.asarray(U_lo_pad, dtype=cp.float32))
    hi_dev = cp.ascontiguousarray(cp.asarray(U_hi_pad, dtype=cp.float32))
    lo_host, hi_host, l0 = canonical._validate_host_intervals(
        cp.asnumpy(lo_dev).reshape(lo_dev.shape),
        cp.asnumpy(hi_dev).reshape(hi_dev.shape),
        l_inf,
    )
    field_v_min, field_v_max, expected_seed_levels = _validate_seeded_host(
        seeds_host, seed_mask, lo_host, hi_host
    )
    Wp, Hp, Dp = (int(v) for v in lo_host.shape)
    N = Wp * Hp * Dp
    lo_f = cp.ascontiguousarray(lo_dev.ravel(), dtype=cp.float32)
    hi_f = cp.ascontiguousarray(hi_dev.ravel(), dtype=cp.float32)
    seeds_f = cp.ascontiguousarray(seeds_dev.ravel(), dtype=cp.float32)

    u = cp.full(N, cp.nan, dtype=cp.float32)
    discovered = cp.zeros(N, dtype=cp.int32)
    processed = cp.zeros(N, dtype=cp.int32)
    scheduled_level = cp.full(N, -1, dtype=cp.int32)
    is_seed = cp.zeros(N, dtype=cp.int32)
    frontier = cp.zeros(N, dtype=cp.int32)
    pending_count = cp.zeros(N_LEVELS, dtype=cp.int32)
    pending_bits = cp.zeros((N_LEVELS + 31) // 32, dtype=cp.uint32)
    activated_count = cp.zeros(1, dtype=cp.int32)
    schedule_dir_count = cp.zeros(3, dtype=cp.int32)
    next_level = cp.full(1, -1, dtype=cp.int32)
    tie_event = cp.zeros(1, dtype=cp.int32)
    has_next = cp.zeros(1, dtype=cp.int32)

    discovered[0] = 1
    scheduled_level[0] = l0
    pending_count[l0] = 1
    pending_bits[l0 >> 5] = cp.uint32(1 << (l0 & 31))

    kernels = _kernel_bundle(cp, canonical._RAW_KERNEL_CODE)
    block = int(block_size)
    grid = (N + block - 1) // block
    kernels["seed_init"](
        (grid,), (block,),
        (
            seeds_f, u, discovered, scheduled_level, is_seed,
            pending_count, pending_bits, field_v_min, field_v_max, N,
        ),
    )
    cp.cuda.Stream.null.synchronize()

    current = l0
    level_transitions: list[list[int]] = []
    closure_rounds_by_level: dict[str, int] = {}
    total_activation_rounds = 0
    backward_scheduler_transitions = 0
    tie_events = 0
    safety_limit = max(10000, N + 10 * N_LEVELS)

    while True:
        rounds = 0
        while _scalar_int(cp, pending_count[current]) != 0:
            rounds += 1
            total_activation_rounds += 1
            if total_activation_rounds > safety_limit:
                raise RuntimeError(
                    f"safety limit exceeded ({safety_limit}) while draining level {current}"
                )
            activated_count[0] = 0
            kernels["activate"](
                (grid,), (block,),
                (
                    u, discovered, processed, scheduled_level, is_seed, frontier,
                    pending_count, pending_bits, activated_count, current, N,
                ),
            )
            cp.cuda.Stream.null.synchronize()
            activated = _scalar_int(cp, activated_count)
            if activated <= 0:
                remaining = _scalar_int(cp, pending_count[current])
                raise RuntimeError(
                    "pending-count invariant failed: "
                    f"level {current} has {remaining} pending items but activated 0"
                )
            kernels["expand"](
                (grid,), (block,),
                (
                    lo_f, hi_f, discovered, processed, scheduled_level, frontier,
                    pending_count, pending_bits, schedule_dir_count,
                    current, Wp, Hp, Dp,
                ),
            )
            cp.cuda.Stream.null.synchronize()

        closure_rounds_by_level[str(current)] = (
            closure_rounds_by_level.get(str(current), 0) + rounds
        )
        next_level[0] = -1
        tie_event[0] = 0
        has_next[0] = 0
        kernels["select"](
            (1,), (1,),
            (
                pending_count, pending_bits, current,
                1 if tie_policy == "higher" else 0,
                next_level, tie_event, has_next,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        if _scalar_int(cp, has_next) == 0:
            break
        nxt = _scalar_int(cp, next_level)
        tie_events += int(bool(_scalar_int(cp, tie_event)))
        backward_scheduler_transitions += int(nxt < current)
        level_transitions.append([int(current), int(nxt)])
        current = nxt

    out = cp.asnumpy(u.reshape((Wp, Hp, Dp)))
    discovered_np = cp.asnumpy(discovered).astype(bool).reshape((Wp, Hp, Dp))
    processed_np = cp.asnumpy(processed).astype(bool).reshape((Wp, Hp, Dp))
    scheduled_np = cp.asnumpy(scheduled_level).reshape((Wp, Hp, Dp))
    is_seed_np = cp.asnumpy(is_seed).astype(bool).reshape((Wp, Hp, Dp))
    pending_count_np = cp.asnumpy(pending_count)
    pending_bits_np = cp.asnumpy(pending_bits)
    dirs = cp.asnumpy(schedule_dir_count).astype(np.int64)

    if int(pending_count_np.sum()) != 0 or bool(np.any(pending_bits_np)):
        raise RuntimeError("worklist terminated with non-empty pending state")
    if not np.array_equal(discovered_np, processed_np):
        raise RuntimeError("worklist terminated with discovered != processed")
    if not np.array_equal(is_seed_np, seed_mask):
        raise RuntimeError("device seed mask differs from validated host seed mask")
    if not np.array_equal(scheduled_np[seed_mask], expected_seed_levels[seed_mask]):
        raise RuntimeError("device seed schedule differs from validated rint/clip schedule")
    raw_preserved = np.array_equal(
        out[seed_mask].view(np.uint32), seeds_host[seed_mask].view(np.uint32)
    )
    if not raw_preserved:
        raise RuntimeError("raw seed bits were not preserved")

    meta: dict[str, Any] = {
        "implementation": "P2_S2C_SEED_AWARE_GPU_WORKLIST_PROTOTYPE",
        "implementation_version": IMPLEMENTATION_VERSION,
        "canonical_native_v1_1_sha256": CANONICAL_NATIVE_V1_1_SHA256,
        "unseeded_delegated_to_canonical": False,
        "bucket_count": N_LEVELS,
        "field_V_MIN": field_v_min,
        "field_V_MAX": field_v_max,
        "l_inf_validated_integer": int(l0),
        "tie_policy": tie_policy,
        "seed_count": int(seed_mask.sum()),
        "seed_activation_count": int(np.count_nonzero(processed_np & is_seed_np)),
        "seed_raw_bits_preserved": bool(raw_preserved),
        "nan_count": int(np.isnan(out).sum()),
        "discovered_count": int(discovered_np.sum()),
        "processed_count": int(processed_np.sum()),
        "discovered_equals_processed": bool(np.array_equal(discovered_np, processed_np)),
        "pending_total_final": int(pending_count_np.sum()),
        "pending_bitset_nonzero_words_final": int(np.count_nonzero(pending_bits_np)),
        "schedule_direction_counts": {
            "backward": int(dirs[0]), "same": int(dirs[1]), "forward": int(dirs[2]),
        },
        "scheduler_transition_count": len(level_transitions),
        "backward_scheduler_transition_count": int(backward_scheduler_transitions),
        "equidistant_next_level_tie_events": int(tie_events),
        "total_activation_rounds": int(total_activation_rounds),
        "closure_rounds_by_level": closure_rounds_by_level,
        "level_transitions": level_transitions,
    }
    aux = {
        "discovered": discovered_np,
        "processed": processed_np,
        "scheduled_level": scheduled_np,
        "seed_injection_level": expected_seed_levels,
        "is_seed": is_seed_np,
        "pending_count": pending_count_np,
        "pending_bits": pending_bits_np,
    }
    return out, meta, aux
