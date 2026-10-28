#!/usr/bin/env python3
"""
step1i_gpu_native_worklist_v1.py
================================

Independent GPU-native prototype for the Step-1I hierarchical worklist.

This working file is the audit-hardened v1.1 revision of the original v1
prototype; the CUDA traversal semantics are unchanged.

This file is deliberately separate from the repaired production
``gpu_immersion.py``.  It implements only the padded, uint8-derived
six-neighbour path used by the Step-1I diagnostic intervention:

* 256 integer level buckets (0..255);
* a device-resident pending count and pending bitset;
* atomic first discovery of a cell;
* one immutable ``scheduled_level[cell]`` assigned at first exposure;
* same-level frontier closure;
* device-side nearest occupied-level selection, including levels below the
  current level.

The intended first gate is exact equality with both the frozen diagnostic
bucket solver and Boutry ``fp_cpu``.  This is not a drop-in replacement for
the repository production kernel and does not implement ``boundary_seeds``.
The current scope is intentionally fail-closed: interval endpoints must be
finite integer-valued float32 values in [0, 255], and flat index 0 is the
sentinel seed used by the Step-1I padded path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


N_LEVELS = 256
V_MIN = 0
V_MAX = 255
DEFAULT_BLOCK = 256
IMPLEMENTATION_VERSION = "1.1-audit-hardened"


_RAW_KERNEL_CODE = r'''
extern "C" {

__device__ __forceinline__ int snap_level(
    int current_level,
    float lo,
    float hi
) {
    float lv = (float)current_level;
    float snapped = (lv < lo) ? lo : ((lv > hi) ? hi : lv);
    // Step-1I endpoints are exact integer-valued float32 values.  Keeping the
    // CUDA round-to-nearest conversion makes the contract explicit and agrees
    // with the diagnostic intervention's __float2int_rn path.
    int target = __float2int_rn(snapped);
    if (target < 0) target = 0;
    if (target > 255) target = 255;
    return target;
}

__global__ void activate_level(
    float* u,
    const int* discovered,
    int* processed,
    const int* scheduled_level,
    int* frontier,
    int* pending_count,
    unsigned int* pending_bits,
    int* activated_count,
    int current_level,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (!discovered[idx]) return;
    if (processed[idx]) return;
    if (scheduled_level[idx] != current_level) return;

    if (atomicCAS(&processed[idx], 0, 1) == 0) {
        u[idx] = (float)current_level;
        frontier[idx] = 1;

        // Activation and expansion are separate kernel launches.  Therefore
        // no increment can race this decrement.  The bit is cleared only when
        // this decrement consumed the last pending item in the bucket.
        int old_count = atomicSub(&pending_count[current_level], 1);
        if (old_count == 1) {
            unsigned int mask = ~(1u << (current_level & 31));
            atomicAnd(&pending_bits[current_level >> 5], mask);
        }
        atomicAdd(activated_count, 1);
    }
}

__global__ void expand_frontier(
    const float* U_lo,
    const float* U_hi,
    int* discovered,
    const int* processed,
    int* scheduled_level,
    int* frontier,
    int* pending_count,
    unsigned int* pending_bits,
    int* schedule_dir_count,
    int current_level,
    int Wp,
    int Hp,
    int Dp
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = Wp * Hp * Dp;
    if (idx >= N) return;
    if (frontier[idx] == 0) return;

    // Each index is owned by exactly one thread in this launch.  The atomic
    // exchange documents the single-use frontier invariant and protects this
    // prototype if the launch geometry is changed later.
    if (atomicExch(&frontier[idx], 0) == 0) return;
    if (!processed[idx]) return;

    int x = idx / (Hp * Dp);
    int rem = idx % (Hp * Dp);
    int y = rem / Dp;
    int z = rem % Dp;

    const int dx[6] = {-1, 1, 0, 0, 0, 0};
    const int dy[6] = {0, 0, -1, 1, 0, 0};
    const int dz[6] = {0, 0, 0, 0, -1, 1};

    for (int d = 0; d < 6; ++d) {
        int nx = x + dx[d];
        int ny = y + dy[d];
        int nz = z + dz[d];
        if (nx < 0 || nx >= Wp || ny < 0 || ny >= Hp || nz < 0 || nz >= Dp)
            continue;

        // C-order flattening for shape (Wp, Hp, Dp).
        int nidx = (nx * Hp + ny) * Dp + nz;

        // First exposure is the only event allowed to assign a schedule.
        // All parents in this launch were activated at the same level, so a
        // winning parent produces the same target as every other parent.
        if (atomicCAS(&discovered[nidx], 0, 1) != 0)
            continue;

        int target = snap_level(current_level, U_lo[nidx], U_hi[nidx]);
        scheduled_level[nidx] = target;
        atomicAdd(&pending_count[target], 1);
        atomicOr(&pending_bits[target >> 5], 1u << (target & 31));

        if (target < current_level) atomicAdd(&schedule_dir_count[0], 1);
        else if (target == current_level) atomicAdd(&schedule_dir_count[1], 1);
        else atomicAdd(&schedule_dir_count[2], 1);
    }
}

__global__ void select_next_level(
    const int* pending_count,
    const unsigned int* pending_bits,
    int current_level,
    int tie_policy,
    int* next_level,
    int* tie_event,
    int* has_next
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int best = -1;
    int best_dist = 0x7fffffff;
    int tied = 0;

    for (int level = 0; level < 256; ++level) {
        unsigned int mask = 1u << (level & 31);
        if ((pending_bits[level >> 5] & mask) == 0u) continue;
        if (pending_count[level] <= 0) continue;

        int dist = (level >= current_level)
            ? (level - current_level)
            : (current_level - level);
        if (dist < best_dist) {
            best = level;
            best_dist = dist;
            tied = 0;
        } else if (dist == best_dist) {
            tied = 1;
            if (tie_policy == 1) {
                if (level > best) best = level;
            } else {
                if (level < best) best = level;
            }
        }
    }

    if (best < 0) {
        *next_level = -1;
        *tie_event = 0;
        *has_next = 0;
    } else {
        *next_level = best;
        *tie_event = tied;
        *has_next = 1;
    }
}

} // extern "C"
'''


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _validate_host_intervals(
    lo: np.ndarray,
    hi: np.ndarray,
    l_inf: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    lo = np.ascontiguousarray(np.asarray(lo, dtype=np.float32))
    hi = np.ascontiguousarray(np.asarray(hi, dtype=np.float32))
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
            "v1 requires integer-valued interval endpoints; "
            f"max fractional deviation={frac}"
        )
    if float(lo.min()) < V_MIN or float(hi.max()) > V_MAX:
        raise ValueError("v1 requires all interval endpoints in [0, 255]")
    l_inf_f = float(l_inf)
    if not math.isfinite(l_inf_f):
        raise ValueError("l_inf must be finite")
    if abs(l_inf_f - round(l_inf_f)) > 1e-7:
        raise ValueError(
            "v1.1 requires integer-valued l_inf; "
            f"got {l_inf_f}"
        )
    if l_inf_f < V_MIN or l_inf_f > V_MAX:
        raise ValueError(
            "v1.1 requires l_inf in [0, 255]; "
            f"got {l_inf_f}"
        )
    l0 = int(round(l_inf_f))
    return lo, hi, l0


def _kernel_bundle(cp):
    module = cp.RawModule(code=_RAW_KERNEL_CODE, options=("-std=c++17",))
    return {
        "activate": module.get_function("activate_level"),
        "expand": module.get_function("expand_frontier"),
        "select": module.get_function("select_next_level"),
    }


def _scalar_int(cp, a) -> int:
    return int(cp.asnumpy(a).reshape(-1)[0])


def _pick_next_host(counts: np.ndarray, current: int, tie_policy: str) -> tuple[int | None, bool]:
    occupied = np.flatnonzero(np.asarray(counts) > 0)
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


def boutry_bucket_gpu(
    cp,
    U_lo_pad,
    U_hi_pad,
    l_inf: float,
    tie_policy: str = "lower",
    block_size: int = DEFAULT_BLOCK,
):
    """Run the independent GPU-native hierarchical worklist v1.

    The host launches kernels and synchronizes between closure rounds, but it
    never performs per-cell discovery, activation, expansion, or level scans.
    The next-level choice is made by the device ``select_next_level`` kernel;
    the host only reads the resulting single level scalar to drive the next
    kernel launch.  This is intentionally a correctness-first v1, not a
    launch-overhead-optimized production implementation.
    """
    if tie_policy not in ("lower", "higher"):
        raise ValueError("tie_policy must be 'lower' or 'higher'")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")

    lo_dev = cp.ascontiguousarray(cp.asarray(U_lo_pad, dtype=cp.float32))
    hi_dev = cp.ascontiguousarray(cp.asarray(U_hi_pad, dtype=cp.float32))
    lo_host, hi_host, l0 = _validate_host_intervals(
        cp.asnumpy(lo_dev).reshape(lo_dev.shape),
        cp.asnumpy(hi_dev).reshape(hi_dev.shape),
        l_inf,
    )
    Wp, Hp, Dp = (int(v) for v in lo_host.shape)
    N = Wp * Hp * Dp
    lo_f = cp.ascontiguousarray(lo_dev.ravel(), dtype=cp.float32)
    hi_f = cp.ascontiguousarray(hi_dev.ravel(), dtype=cp.float32)

    u = cp.full(N, cp.nan, dtype=cp.float32)
    discovered = cp.zeros(N, dtype=cp.int32)
    processed = cp.zeros(N, dtype=cp.int32)
    scheduled_level = cp.full(N, -1, dtype=cp.int32)
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

    kernels = _kernel_bundle(cp)
    block = int(block_size)
    grid = (N + block - 1) // block
    current = l0
    level_transitions: list[list[int]] = []
    closure_rounds_by_level: dict[str, int] = {}
    total_activation_rounds = 0
    backward_scheduler_transitions = 0
    tie_events = 0
    safety_limit = max(10000, N + 10 * N_LEVELS)

    while True:
        rounds = 0
        while True:
            current_pending = _scalar_int(cp, pending_count[current])
            if current_pending == 0:
                break

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
                    u,
                    discovered,
                    processed,
                    scheduled_level,
                    frontier,
                    pending_count,
                    pending_bits,
                    activated_count,
                    current,
                    N,
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
                    lo_f,
                    hi_f,
                    discovered,
                    processed,
                    scheduled_level,
                    frontier,
                    pending_count,
                    pending_bits,
                    schedule_dir_count,
                    current,
                    Wp,
                    Hp,
                    Dp,
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
                pending_count,
                pending_bits,
                current,
                1 if tie_policy == "higher" else 0,
                next_level,
                tie_event,
                has_next,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        if _scalar_int(cp, has_next) == 0:
            break

        nxt = _scalar_int(cp, next_level)
        if _scalar_int(cp, tie_event):
            tie_events += 1
        if nxt < current:
            backward_scheduler_transitions += 1
        level_transitions.append([int(current), int(nxt)])
        current = nxt

    out = cp.asnumpy(u.reshape((Wp, Hp, Dp)))
    discovered_np = cp.asnumpy(discovered).astype(bool).reshape((Wp, Hp, Dp))
    processed_np = cp.asnumpy(processed).astype(bool).reshape((Wp, Hp, Dp))
    scheduled_np = cp.asnumpy(scheduled_level).reshape((Wp, Hp, Dp))
    pending_count_np = cp.asnumpy(pending_count)
    pending_bits_np = cp.asnumpy(pending_bits)
    dirs = cp.asnumpy(schedule_dir_count).astype(np.int64)

    if int(pending_count_np.sum()) != 0 or bool(np.any(pending_bits_np)):
        raise RuntimeError("worklist terminated with non-empty pending state")

    meta: dict[str, Any] = {
        "implementation": "GPU_NATIVE_WORKLIST_V1",
        "implementation_version": IMPLEMENTATION_VERSION,
        "bucket_count": N_LEVELS,
        "V_MIN": V_MIN,
        "V_MAX": V_MAX,
        "l_inf_validated_integer": int(l0),
        "tie_policy": tie_policy,
        "host_per_cell_work": False,
        "device_next_level_selection": True,
        "host_current_bucket_polling": True,
        "nan_count": int(np.isnan(out).sum()),
        "discovered_count": int(discovered_np.sum()),
        "processed_count": int(processed_np.sum()),
        "pending_total_final": int(pending_count_np.sum()),
        "pending_bitset_nonzero_words_final": int(np.count_nonzero(pending_bits_np)),
        "schedule_direction_counts": {
            "backward": int(dirs[0]),
            "same": int(dirs[1]),
            "forward": int(dirs[2]),
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
        "pending_count": pending_count_np,
        "pending_bits": pending_bits_np,
    }
    return out, meta, aux


def _neighbors(idx: int, Wp: int, Hp: int, Dp: int):
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


def cpu_shadow_worklist_v1(
    U_lo_pad: np.ndarray,
    U_hi_pad: np.ndarray,
    l_inf: float,
    tie_policy: str = "lower",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Small CPU shadow of the v1 state machine for no-CUDA unit tests.

    This is not used as the semantic oracle.  It exists to exercise the same
    state transitions and invariants before a CUDA host is available.
    """
    lo, hi, l0 = _validate_host_intervals(U_lo_pad, U_hi_pad, l_inf)
    Wp, Hp, Dp = lo.shape
    N = lo.size
    lo_f, hi_f = lo.ravel(), hi.ravel()
    out = np.full(N, np.nan, dtype=np.float32)
    discovered = np.zeros(N, dtype=bool)
    processed = np.zeros(N, dtype=bool)
    scheduled = np.full(N, -1, dtype=np.int32)
    pending: list[list[int]] = [[] for _ in range(N_LEVELS)]
    pending_count = np.zeros(N_LEVELS, dtype=np.int64)

    discovered[0] = True
    scheduled[0] = l0
    pending[l0].append(0)
    pending_count[l0] += 1

    dirs = np.zeros(3, dtype=np.int64)
    transitions: list[list[int]] = []
    tie_events = 0
    activation_rounds = 0
    backward_transitions = 0
    current = l0

    while True:
        while pending_count[current] > 0:
            activation_rounds += 1
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
                out[idx] = np.float32(current)
                frontier.append(idx)

            # The GPU version clears the frontier in one kernel and performs
            # all claims in the next kernel.  Preserve that two-phase order.
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
                        dirs[0] += 1
                    elif target == current:
                        dirs[1] += 1
                    else:
                        dirs[2] += 1

        nxt, tied = _pick_next_host(pending_count, current, tie_policy)
        if nxt is None:
            break
        tie_events += int(tied)
        if nxt < current:
            backward_transitions += 1
        transitions.append([int(current), int(nxt)])
        current = nxt

    if np.any(pending_count != 0):
        raise RuntimeError("CPU shadow terminated with pending work")
    if not np.all(discovered == processed):
        raise RuntimeError("CPU shadow has discovered-but-unprocessed cells")

    return out.reshape(lo.shape), {
        "implementation": "CPU_SHADOW_ONLY",
        "nan_count": int(np.isnan(out).sum()),
        "discovered_count": int(discovered.sum()),
        "processed_count": int(processed.sum()),
        "schedule_direction_counts": {
            "backward": int(dirs[0]),
            "same": int(dirs[1]),
            "forward": int(dirs[2]),
        },
        "backward_scheduler_transition_count": int(backward_transitions),
        "equidistant_next_level_tie_events": int(tie_events),
        "total_activation_rounds": int(activation_rounds),
        "level_transitions": transitions,
    }


def run_core_gpu_fixture(cp) -> dict[str, Any]:
    """The canonical 0 -> 5 -> 3 GPU-native smoke fixture."""
    lo = np.array([[[0, 5, 0]]], dtype=np.float32)
    hi = np.array([[[0, 5, 3]]], dtype=np.float32)
    out, meta, _ = boutry_bucket_gpu(cp, cp.asarray(lo), cp.asarray(hi), 0.0)
    expected = np.array([[[0, 5, 3]]], dtype=np.float32)
    if not np.array_equal(out, expected):
        raise RuntimeError(f"GPU_NATIVE_CORE_FIXTURE_FAIL: {out.tolist()}")
    if meta["backward_scheduler_transition_count"] < 1:
        raise RuntimeError("GPU_NATIVE_CORE_FIXTURE_FAIL: no backward transition")
    return {
        "pass": True,
        "output": out.tolist(),
        "expected": expected.tolist(),
        "backward_scheduler_transition_count": meta["backward_scheduler_transition_count"],
        "schedule_direction_counts": meta["schedule_direction_counts"],
        "level_transitions": meta["level_transitions"],
    }


def cpu_shadow_self_test() -> dict[str, Any]:
    """Run deterministic no-CUDA state-machine checks."""
    cases = []
    fixtures = [
        (
            "backward_revisit_0_5_3",
            np.array([[[0, 5, 0]]], dtype=np.float32),
            np.array([[[0, 5, 3]]], dtype=np.float32),
            np.array([[[0, 5, 3]]], dtype=np.float32),
        ),
        (
            "pure_forward",
            np.array([[[0, 10, 20, 30]]], dtype=np.float32),
            np.array([[[0, 19, 29, 39]]], dtype=np.float32),
            np.array([[[0, 10, 20, 30]]], dtype=np.float32),
        ),
        (
            "same_level_closure",
            np.array([[[0, 100, 90, 100]]], dtype=np.float32),
            np.array([[[0, 100, 110, 100]]], dtype=np.float32),
            np.array([[[0, 100, 100, 100]]], dtype=np.float32),
        ),
    ]
    for name, lo, hi, expected in fixtures:
        out, meta = cpu_shadow_worklist_v1(lo, hi, 0.0)
        ok = bool(np.array_equal(out, expected))
        if not ok:
            raise AssertionError(f"{name}: got {out.tolist()}, expected {expected.tolist()}")
        cases.append(
            {
                "name": name,
                "pass": True,
                "backward_scheduler_transition_count": meta[
                    "backward_scheduler_transition_count"
                ],
            }
        )
    rejected_l_inf = []
    probe_lo = np.array([[[0, 5, 0]]], dtype=np.float32)
    probe_hi = np.array([[[0, 5, 3]]], dtype=np.float32)
    for bad_l_inf in (0.5, -1.0, 256.0):
        try:
            cpu_shadow_worklist_v1(probe_lo, probe_hi, bad_l_inf)
        except ValueError:
            rejected_l_inf.append(bad_l_inf)
        else:
            raise AssertionError(f"non-fail-closed l_inf accepted: {bad_l_inf}")
    return {
        "verdict": "CPU_SHADOW_SELF_TEST_PASS",
        "case_count": len(cases),
        "cases": cases,
        "strict_l_inf_rejections": rejected_l_inf,
        "scope": "CPU shadow only; no GPU execution",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="Run no-CUDA CPU shadow checks")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not args.self_test:
        ap.error("v1 module is a library; use --self-test or import boutry_bucket_gpu()")
    result = cpu_shadow_self_test()
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
