#!/usr/bin/env python3
"""Seed-aware CPU reference for DD-FP P2-S2a.

This is a semantic reference, not a performance implementation and not a CUDA
or production patch.  It extends the canonical native-v1.1 closest-occupied-
bucket state machine with pre-confirmed immutable boundary seeds while keeping
the unseeded path equivalent to the canonical v1.1 CPU shadow.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np


N_LEVELS = 256
V_MIN = 0
V_MAX = 255
IMPLEMENTATION_VERSION = "P2-S2a-seed-aware-cpu-reference-v1+production-right-ghost-z-v1"
CANONICAL_NATIVE_V1_1_SHA256 = (
    "853bf33a6bbd9c91f5304a018f0a51ff53e8418360fe1477208cf81df17163ac"
)
P2_S1_CONTRACT_ZIP_SHA256 = (
    "7361c3baab24a92c23f75a74357597cc9b51a085b69e91b91a0d0f90276d425f"
)


def _validate_intervals(
    U_lo_pad: np.ndarray,
    U_hi_pad: np.ndarray,
    l_inf: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mirror canonical v1.1 fail-closed interval/l_inf validation."""
    lo = np.ascontiguousarray(np.asarray(U_lo_pad, dtype=np.float32))
    hi = np.ascontiguousarray(np.asarray(U_hi_pad, dtype=np.float32))
    if lo.shape != hi.shape or lo.ndim != 3 or lo.size == 0:
        raise ValueError("U_lo/U_hi must be non-empty, same-shape 3D arrays")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("U_lo/U_hi must be finite")
    if np.any(lo > hi):
        raise ValueError("U_lo must be <= U_hi elementwise")
    frac = max(
        float(np.max(np.abs(lo.astype(np.float64) - np.rint(lo.astype(np.float64))))),
        float(np.max(np.abs(hi.astype(np.float64) - np.rint(hi.astype(np.float64))))),
    )
    if frac > 1e-7:
        raise ValueError(
            "v1.1 requires integer-valued interval endpoints; "
            f"max fractional deviation={frac}"
        )
    if float(lo.min()) < V_MIN or float(hi.max()) > V_MAX:
        raise ValueError("v1.1 requires all interval endpoints in [0, 255]")
    l_inf_f = float(l_inf)
    if not math.isfinite(l_inf_f):
        raise ValueError("l_inf must be finite")
    if abs(l_inf_f - round(l_inf_f)) > 1e-7:
        raise ValueError(f"v1.1 requires integer-valued l_inf; got {l_inf_f}")
    if l_inf_f < V_MIN or l_inf_f > V_MAX:
        raise ValueError(f"v1.1 requires l_inf in [0, 255]; got {l_inf_f}")
    return lo, hi, int(round(l_inf_f))


def _outer_padding_mask(shape: tuple[int, int, int]) -> np.ndarray:
    outer = np.zeros(shape, dtype=bool)
    outer[0, :, :] = True
    outer[-1, :, :] = True
    outer[:, 0, :] = True
    outer[:, -1, :] = True
    outer[:, :, 0] = True
    outer[:, :, -1] = True
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


def _validate_boundary_seeds(
    boundary_seeds: np.ndarray | None,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the P2-S1 paper-path contract and return seed state.

    The returned seed values are the exact float32 values that are frozen in
    ``u``.  Their scheduler levels are derived independently with rint+clip.
    """
    if boundary_seeds is None:
        seeds = np.full(lo.shape, np.nan, dtype=np.float32)
        return seeds, np.zeros(lo.shape, dtype=bool), np.full(lo.shape, -1, np.int32)

    seeds = np.ascontiguousarray(np.asarray(boundary_seeds, dtype=np.float32))
    if seeds.shape != lo.shape or seeds.ndim != 3:
        raise ValueError("boundary_seeds must have the same padded 3-D shape as U_lo/U_hi")

    seed_mask = ~np.isnan(seeds)
    levels = np.full(lo.shape, -1, dtype=np.int32)
    if not np.any(seed_mask):
        return seeds, seed_mask, levels

    values = seeds[seed_mask]
    if not np.all(np.isfinite(values)):
        raise ValueError("all non-NaN boundary seeds must be finite")
    frac = float(np.max(np.abs(values.astype(np.float64) - np.rint(values.astype(np.float64)))))
    if frac > 1e-7:
        raise ValueError(
            "P2-S2 paper path requires integer-valued boundary seeds; "
            f"max fractional deviation={frac}"
        )
    if float(values.min()) < V_MIN or float(values.max()) > V_MAX:
        raise ValueError("P2-S2 paper path requires boundary seeds in [0,255]")
    _validate_production_outer_seed_extension(seed_mask)
    if not np.array_equal(values, lo[seed_mask]) or not np.array_equal(values, hi[seed_mask]):
        raise ValueError("every paper-path seed must equal both pinned singleton endpoints")

    levels[seed_mask] = np.clip(np.rint(values), V_MIN, V_MAX).astype(np.int32)
    return seeds, seed_mask, levels


def _neighbors(idx: int, Wp: int, Hp: int, Dp: int) -> Iterator[int]:
    """Six-neighbour C-order adjacency used by canonical native v1.1."""
    x = idx // (Hp * Dp)
    rem = idx % (Hp * Dp)
    y = rem // Dp
    z = rem % Dp
    if x > 0:
        yield idx - Hp * Dp
    if x + 1 < Wp:
        yield idx + Hp * Dp
    if y > 0:
        yield idx - Dp
    if y + 1 < Hp:
        yield idx + Dp
    if z > 0:
        yield idx - 1
    if z + 1 < Dp:
        yield idx + 1


def _pick_next(
    pending_count: np.ndarray,
    current: int,
    tie_policy: str,
) -> tuple[int | None, bool]:
    occupied = np.flatnonzero(np.asarray(pending_count) > 0)
    if occupied.size == 0:
        return None, False
    distances = np.abs(occupied - current)
    best_distance = int(distances.min())
    candidates = occupied[distances == best_distance]
    if tie_policy == "higher":
        return int(candidates.max()), bool(candidates.size > 1)
    if tie_policy == "lower":
        return int(candidates.min()), bool(candidates.size > 1)
    raise ValueError("tie_policy must be 'lower' or 'higher'")


def seed_aware_cpu_reference(
    U_lo_pad: np.ndarray,
    U_hi_pad: np.ndarray,
    l_inf: float,
    boundary_seeds: np.ndarray | None = None,
    tie_policy: str = "lower",
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Run the P2-S2 seed-aware native reference state machine.

    Seeds are discovered and their raw float32 outputs are frozen at
    initialization, but they propagate only when their scheduled seed bucket is
    activated.  Free cells retain canonical v1.1 first-claim semantics.
    """
    if tie_policy not in ("lower", "higher"):
        raise ValueError("tie_policy must be 'lower' or 'higher'")

    lo, hi, l0 = _validate_intervals(U_lo_pad, U_hi_pad, l_inf)
    seeds, seed_mask_3d, seed_levels_3d = _validate_boundary_seeds(boundary_seeds, lo, hi)
    Wp, Hp, Dp = lo.shape
    N = lo.size
    lo_f, hi_f = lo.ravel(), hi.ravel()
    seed_values = seeds.ravel()
    is_seed = seed_mask_3d.ravel()
    seed_levels = seed_levels_3d.ravel()

    out = np.full(N, np.nan, dtype=np.float32)
    discovered = np.zeros(N, dtype=bool)
    processed = np.zeros(N, dtype=bool)
    scheduled = np.full(N, -1, dtype=np.int32)
    pending: list[list[int]] = [[] for _ in range(N_LEVELS)]
    pending_count = np.zeros(N_LEVELS, dtype=np.int64)

    # Canonical sentinel root.
    discovered[0] = True
    scheduled[0] = l0
    pending[l0].append(0)
    pending_count[l0] += 1

    # Pre-confirmed roots.  Padding validation guarantees idx 0 is not a seed.
    for idx in np.flatnonzero(is_seed):
        level = int(seed_levels[idx])
        out[idx] = seed_values[idx]          # raw float32 value frozen now
        discovered[idx] = True               # ordinary claims lose from time zero
        scheduled[idx] = level
        pending[level].append(int(idx))       # propagation waits for this bucket
        pending_count[level] += 1

    direction_counts = np.zeros(3, dtype=np.int64)
    transitions: list[list[int]] = []
    tie_events = 0
    activation_rounds = 0
    backward_transitions = 0
    seed_activations = 0
    current = l0
    safety_limit = max(10000, N + 10 * N_LEVELS)

    while True:
        while pending_count[current] > 0:
            activation_rounds += 1
            if activation_rounds > safety_limit:
                raise RuntimeError(f"safety limit exceeded while draining level {current}")

            frontier: list[int] = []
            bucket = pending[current]
            cursor = 0
            while cursor < len(bucket):
                idx = bucket[cursor]
                cursor += 1
                if not discovered[idx] or processed[idx] or scheduled[idx] != current:
                    continue
                processed[idx] = True
                pending_count[current] -= 1
                if is_seed[idx]:
                    # The seed output and scheduler level are deliberately
                    # separate states.  Activation must not rewrite raw u[idx].
                    seed_activations += 1
                else:
                    out[idx] = np.float32(current)
                frontier.append(idx)

            # Two-phase order: activate all current work, then expose neighbours.
            for idx in frontier:
                for nb in _neighbors(idx, Wp, Hp, Dp):
                    if discovered[nb]:
                        continue
                    discovered[nb] = True
                    lv = float(current)
                    lov, hiv = float(lo_f[nb]), float(hi_f[nb])
                    snapped = lov if lv < lov else (hiv if lv > hiv else lv)
                    target = int(round(snapped))
                    target = max(V_MIN, min(V_MAX, target))
                    scheduled[nb] = target
                    pending[target].append(nb)
                    pending_count[target] += 1
                    if target < current:
                        direction_counts[0] += 1
                    elif target == current:
                        direction_counts[1] += 1
                    else:
                        direction_counts[2] += 1

        nxt, tied = _pick_next(pending_count, current, tie_policy)
        if nxt is None:
            break
        tie_events += int(tied)
        if nxt < current:
            backward_transitions += 1
        transitions.append([int(current), int(nxt)])
        current = nxt

    if np.any(pending_count != 0):
        raise RuntimeError("seed-aware reference terminated with pending work")
    if not np.array_equal(discovered, processed):
        raise RuntimeError("seed-aware reference has discovered-but-unprocessed cells")
    if np.isnan(out).any():
        raise RuntimeError("seed-aware reference terminated with NaN output")
    seed_bits_preserved = bool(np.array_equal(
        out[is_seed].view(np.uint32), seed_values[is_seed].view(np.uint32)
    ))
    if not seed_bits_preserved:
        raise RuntimeError("raw boundary-seed bits changed during activation")
    if seed_activations != int(is_seed.sum()):
        raise RuntimeError("not every seed was activated exactly once")

    out_3d = out.reshape(lo.shape)
    discovered_3d = discovered.reshape(lo.shape)
    processed_3d = processed.reshape(lo.shape)
    scheduled_3d = scheduled.reshape(lo.shape)
    is_seed_3d = is_seed.reshape(lo.shape)

    meta: dict[str, Any] = {
        "implementation": "SEED_AWARE_CPU_REFERENCE",
        "implementation_version": IMPLEMENTATION_VERSION,
        "canonical_native_v1_1_sha256": CANONICAL_NATIVE_V1_1_SHA256,
        "p2_s1_contract_zip_sha256": P2_S1_CONTRACT_ZIP_SHA256,
        "tie_policy": tie_policy,
        "l_inf_validated_integer": int(l0),
        "seed_count": int(is_seed.sum()),
        "seed_activation_count": int(seed_activations),
        "seed_raw_bits_preserved": seed_bits_preserved,
        "nan_count": int(np.isnan(out).sum()),
        "discovered_count": int(discovered.sum()),
        "processed_count": int(processed.sum()),
        "discovered_equals_processed": bool(np.array_equal(discovered, processed)),
        "pending_total_final": int(pending_count.sum()),
        "schedule_direction_counts": {
            "backward": int(direction_counts[0]),
            "same": int(direction_counts[1]),
            "forward": int(direction_counts[2]),
        },
        "backward_scheduler_transition_count": int(backward_transitions),
        "equidistant_next_level_tie_events": int(tie_events),
        "total_activation_rounds": int(activation_rounds),
        "level_transitions": transitions,
    }
    aux = {
        "discovered": discovered_3d,
        "processed": processed_3d,
        "scheduled_level": scheduled_3d,
        "is_seed": is_seed_3d,
        "seed_value": seeds,
        "seed_injection_level": seed_levels_3d,
        "pending_count": pending_count,
    }
    return out_3d, meta, aux


def source_path() -> Path:
    return Path(__file__).resolve()


if __name__ == "__main__":
    # Keep the reference module side-effect free.  The adjacent test driver owns
    # executable fixtures and machine-readable result generation.
    print(f"{IMPLEMENTATION_VERSION}: import this module or run p2s2_seed_aware_cpu_reference_tests.py")
