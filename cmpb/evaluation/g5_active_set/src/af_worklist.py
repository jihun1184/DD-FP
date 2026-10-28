#!/usr/bin/env python3
"""
af_worklist.py -- K=1 active-frontier prototype (A1), isolated from production.

Scope  : unseeded canonical path only (flat index 0 is the sentinel seed), tie policy 'lower' only.
         No higher tie, no B1/B2, no geodesic reformulation, no IBI seed path.
Base   : frozen ``gpu_native_worklist.py`` v1.1 (SHA-256 853bf33a...).  Its kernel text is compiled
         *byte-for-byte* into the same module and is used (a) for ``select_next_level`` and (b) as the
         state-preserving fallback (``activate_level`` / ``expand_frontier``).
Idea   : the frozen host loop launches ``activate_level`` and ``expand_frontier`` over all N cells in every
         closure round.  Here the set pending at the start of a level visit is gathered once (full scan),
         and each round then processes a compact list: one fused kernel activates the listed cells and
         expands their 6-neighbours; cells first-discovered at the *current* level go to the next list.
Round boundary (the thing that must be preserved): a cell discovered during round r at the current level is
         processed in round r+1, never in round r.  The fused kernel reads only ``list_cur`` and appends only
         to ``list_next``; the two lists are swapped between launches (kernel-launch boundary == barrier).
State  : ``discovered``, ``scheduled_level``, ``processed``, ``u``, ``pending_count``, ``pending_bits``,
         ``schedule_dir_count`` are the *authoritative* state and are updated exactly as in the frozen code
         at every round boundary; the lists are a cache of {discovered & !processed & scheduled_level==c}.
Overflow: ``list_next`` writes are guarded by ``pos < cap``.  If the true count exceeds ``cap`` the arrays
         already equal the frozen post-round state (``pending_count[c]`` is written with the *unclamped*
         count), the list is dropped and the visit continues with the frozen kernels.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

FROZEN_SHA256 = "853bf33a6bbd9c91f5304a018f0a51ff53e8418360fe1477208cf81df17163ac"
IMPLEMENTATION = "AF_ACTIVE_FRONTIER_K1_V0"
DEFAULT_CAP = 4_194_304  # 4 Mi cells -> 2 lists x 16 MiB

AF_NEW_CODE = r'''
extern "C" {

// ---------------------------------------------------------------------------------------------
// af_gather: list {discovered && !processed && scheduled_level == level}.
// The predicate is the conjunction used by activate_level; only the order of the three reads
// differs (scheduled_level first: it rejects almost every cell with one 4-byte read).
// All N threads run; `counter` counts *all* matches, the list stores at most `cap` of them.
// ---------------------------------------------------------------------------------------------
__global__ void af_gather(
    const int* discovered,
    const int* processed,
    const int* scheduled_level,
    int* list_out,
    int* counter,
    int level,
    int cap,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (scheduled_level[idx] != level) return;
    if (processed[idx]) return;
    if (!discovered[idx]) return;
    int pos = atomicAdd(counter, 1);
    if (pos < cap) list_out[pos] = idx;
}

// ---------------------------------------------------------------------------------------------
// af_round: one closure round == activate_level + expand_frontier of the frozen code, fused.
//   activation : u = level, processed = 1                      (cells of list_cur only)
//   expansion  : atomicCAS(discovered) first-discovery claim, scheduled_level = snap_level(...)
//   same level : appended to list_next (processed in the NEXT launch), pending_count[level]
//                and pending_bits[level] are written by af_finalize, not here
//   other level: pending_count / pending_bits updated exactly as in expand_frontier
// ---------------------------------------------------------------------------------------------
__global__ void af_round(
    float* u,
    int* discovered,
    int* processed,
    int* scheduled_level,
    const float* U_lo,
    const float* U_hi,
    int* pending_count,
    unsigned int* pending_bits,
    int* schedule_dir_count,
    const int* list_cur,
    int n_cur,
    int* list_next,
    int* counter_next,
    int cap,
    int current_level,
    int Wp,
    int Hp,
    int Dp
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cur || i >= cap) return;
    int idx = list_cur[i];

    u[idx] = (float)current_level;
    processed[idx] = 1;

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
        int nidx = (nx * Hp + ny) * Dp + nz;

        if (atomicCAS(&discovered[nidx], 0, 1) != 0)
            continue;

        int target = snap_level(current_level, U_lo[nidx], U_hi[nidx]);
        scheduled_level[nidx] = target;

        if (target == current_level) {
            int pos = atomicAdd(counter_next, 1);
            if (pos < cap) list_next[pos] = nidx;
            atomicAdd(&schedule_dir_count[1], 1);
        } else {
            atomicAdd(&pending_count[target], 1);
            atomicOr(&pending_bits[target >> 5], 1u << (target & 31));
            if (target < current_level) atomicAdd(&schedule_dir_count[0], 1);
            else atomicAdd(&schedule_dir_count[2], 1);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// af_finalize: single thread, runs after af_gather (mode 0) or af_round (mode 1) on the same stream.
//   out[0] = total count (unclamped)   out[1] = total > cap   out[2] = mismatch (mode 0 only)
//   mode 1 : every cell that was pending at `level` has been consumed by af_round, so the new pending
//            count of `level` is exactly the number of same-level first discoveries of that round.
//            The occupancy bit follows the count (frozen code: cleared by the last decrement, set again
//            by any same-level increment of expand_frontier).
//   mode 0 : cross-check the gathered count against the authoritative pending_count[level].
// ---------------------------------------------------------------------------------------------
__global__ void af_finalize(
    int* counter,
    int* pending_count,
    unsigned int* pending_bits,
    int* out,
    int level,
    int mode,
    int cap
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int total = *counter;
    *counter = 0;
    out[0] = total;
    out[1] = (total > cap) ? 1 : 0;
    out[2] = 0;
    if (mode == 1) {
        pending_count[level] = total;
        unsigned int bit = 1u << (level & 31);
        if (total > 0) atomicOr(&pending_bits[level >> 5], bit);
        else atomicAnd(&pending_bits[level >> 5], ~bit);
    } else {
        out[2] = (total != pending_count[level]) ? 1 : 0;
    }
}

} // extern "C"
'''


# --------------------------------------------------------------------------------------------------
def frozen_path() -> Path:
    return Path(__file__).resolve().parent.parent / "frozen" / "gpu_native_worklist.py"


def load_frozen(path: Path | None = None):
    path = Path(path) if path else frozen_path()
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != FROZEN_SHA256:
        raise RuntimeError(f"frozen worklist hash mismatch: {h}")
    spec = importlib.util.spec_from_file_location("_af_frozen_native_v1_1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def module_code(frozen, new_code: str | None = None) -> str:
    """frozen kernel text (verbatim) followed by the new kernels; snap_level is shared, not copied."""
    return frozen._RAW_KERNEL_CODE + "\n" + (AF_NEW_CODE if new_code is None else new_code)


def _scalar_int(cp, a) -> int:
    return int(cp.asnumpy(a).reshape(-1)[0])


class _Prof:
    """Optional breakdown. GPU time by events; host time by perf_counter. Never changes semantics."""

    def __init__(self, cp, on: bool):
        self.cp, self.on = cp, on
        self.host: dict[str, float] = {}
        self.gpu_ms: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self._ev: list[tuple[str, Any, Any]] = []

    def add_host(self, name: str, dt: float):
        self.host[name] = self.host.get(name, 0.0) + dt
        self.calls[name] = self.calls.get(name, 0) + 1

    def begin(self):
        if not self.on:
            return None
        e = self.cp.cuda.Event()
        e.record()
        return (time.perf_counter(), e)

    def end(self, name: str, tok):
        if tok is None:
            return
        t0, e0 = tok
        e1 = self.cp.cuda.Event()
        e1.record()
        self._ev.append((name, e0, e1))
        self.add_host("launch:" + name, time.perf_counter() - t0)

    def resolve(self):
        if not self.on:
            return
        for name, e0, e1 in self._ev:
            e1.synchronize()
            self.gpu_ms[name] = self.gpu_ms.get(name, 0.0) + float(self.cp.cuda.get_elapsed_time(e0, e1))
        self._ev.clear()


def boutry_active_frontier(
    cp,
    U_lo_pad,
    U_hi_pad,
    l_inf: float,
    tie_policy: str = "lower",
    block_size: int = 256,
    frontier_cap: int | None = None,
    *,
    frozen=None,
    new_code: str | None = None,
    round_hook: Callable[[dict], None] | None = None,
    profile: bool = False,
    guard: int = 0,
    trace: bool = False,
):
    """Active-frontier K=1 worklist.  Returns (out, meta, aux) like ``boutry_bucket_gpu``.

    guard   : test-only. Allocates ``guard`` extra int32 slots behind each list, fills them with a sentinel and
              verifies after every list-writing launch that nothing was written there (out-of-bounds canary).
    trace   : record per-round frontier sizes and per-visit closure depth in ``meta['af']``.
    """
    if tie_policy != "lower":
        raise ValueError("A1 prototype supports tie_policy='lower' only")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    F = frozen or load_frozen()
    N_LEVELS = F.N_LEVELS
    prof = _Prof(cp, profile)
    t_total0 = time.perf_counter()

    lo_dev = cp.ascontiguousarray(cp.asarray(U_lo_pad, dtype=cp.float32))
    hi_dev = cp.ascontiguousarray(cp.asarray(U_hi_pad, dtype=cp.float32))
    lo_host, hi_host, l0 = F._validate_host_intervals(
        cp.asnumpy(lo_dev).reshape(lo_dev.shape),
        cp.asnumpy(hi_dev).reshape(hi_dev.shape),
        l_inf,
    )
    Wp, Hp, Dp = (int(v) for v in lo_host.shape)
    N = Wp * Hp * Dp
    if N >= 2**31 - 1 or (Wp * Hp * Dp) != N:
        raise ValueError("grid too large for int32 indexing")
    lo_f = cp.ascontiguousarray(lo_dev.ravel(), dtype=cp.float32)
    hi_f = cp.ascontiguousarray(hi_dev.ravel(), dtype=cp.float32)

    cap = int(min(N, DEFAULT_CAP if frontier_cap is None else frontier_cap))
    if cap < 1:
        raise ValueError("frontier_cap must be >= 1")

    u = cp.full(N, cp.nan, dtype=cp.float32)
    discovered = cp.zeros(N, dtype=cp.int32)
    processed = cp.zeros(N, dtype=cp.int32)
    scheduled_level = cp.full(N, -1, dtype=cp.int32)
    frontier = cp.zeros(N, dtype=cp.int32)  # used only by the frozen fallback kernels; stays all-zero otherwise
    pending_count = cp.zeros(N_LEVELS, dtype=cp.int32)
    pending_bits = cp.zeros((N_LEVELS + 31) // 32, dtype=cp.uint32)
    activated_count = cp.zeros(1, dtype=cp.int32)
    schedule_dir_count = cp.zeros(3, dtype=cp.int32)
    next_level = cp.full(1, -1, dtype=cp.int32)
    tie_event = cp.zeros(1, dtype=cp.int32)
    has_next = cp.zeros(1, dtype=cp.int32)
    # active-frontier state
    SENT = np.int32(-0x5A5A5A5)
    list_a = cp.zeros(cap + guard, dtype=cp.int32)
    list_b = cp.zeros(cap + guard, dtype=cp.int32)
    if guard:
        list_a[cap:] = SENT
        list_b[cap:] = SENT
    counter = cp.zeros(1, dtype=cp.int32)
    out_dev = cp.zeros(4, dtype=cp.int32)

    discovered[0] = 1
    scheduled_level[0] = l0
    pending_count[l0] = 1
    pending_bits[l0 >> 5] = cp.uint32(1 << (l0 & 31))

    module = cp.RawModule(code=module_code(F, new_code), options=("-std=c++17",))
    K = {n: module.get_function(n) for n in
         ("activate_level", "expand_frontier", "select_next_level", "af_gather", "af_round", "af_finalize")}
    block = int(block_size)
    grid_full = (N + block - 1) // block
    i32 = np.int32

    arrays = dict(u=u, discovered=discovered, processed=processed, scheduled_level=scheduled_level,
                  pending_count=pending_count, pending_bits=pending_bits,
                  schedule_dir_count=schedule_dir_count, frontier=frontier)

    def d2h(a):
        return a.get() if hasattr(a, "get") else cp.asnumpy(a)

    def check_guard():
        if guard:
            for nm, L in (("list_a", list_a), ("list_b", list_b)):
                if not np.all(d2h(L)[cap:] == SENT):
                    raise RuntimeError(f"out-of-bounds write behind {nm} (guard canary overwritten)")

    level_transitions: list[list[int]] = []
    closure_rounds_by_level: dict[str, int] = {}
    total_activation_rounds = 0
    backward_scheduler_transitions = 0
    tie_events = 0
    safety_limit = max(10000, N + 10 * N_LEVELS)
    st = dict(rounds_list=0, rounds_frozen=0, visits=0, visits_list=0, visits_gather_overflow=0,
              visits_mid_fallback=0, gather_calls=0, max_frontier=0)
    frontier_sizes: list[int] = []
    visit_depths: list[int] = []

    def frozen_round(current: int) -> None:
        activated_count[0] = 0
        tk = prof.begin()
        K["activate_level"]((grid_full,), (block,), (u, discovered, processed, scheduled_level, frontier,
                                                      pending_count, pending_bits, activated_count, current, N))
        prof.end("frozen_activate", tk)
        t = time.perf_counter(); cp.cuda.Stream.null.synchronize(); prof.add_host("sync", time.perf_counter() - t)
        t = time.perf_counter(); activated = _scalar_int(cp, activated_count); prof.add_host("d2h", time.perf_counter() - t)
        if activated <= 0:
            remaining = _scalar_int(cp, pending_count[current])
            raise RuntimeError("pending-count invariant failed: "
                               f"level {current} has {remaining} pending items but activated 0")
        tk = prof.begin()
        K["expand_frontier"]((grid_full,), (block,), (lo_f, hi_f, discovered, processed, scheduled_level, frontier,
                                                       pending_count, pending_bits, schedule_dir_count,
                                                       current, Wp, Hp, Dp))
        prof.end("frozen_expand", tk)
        t = time.perf_counter(); cp.cuda.Stream.null.synchronize(); prof.add_host("sync", time.perf_counter() - t)
        if trace:
            frontier_sizes.append(activated)

    current = l0
    while True:
        rounds = 0
        st["visits"] += 1
        # ------------------------------ visit start: gather the pending set of `current` -------------
        tk = prof.begin()
        K["af_gather"]((grid_full,), (block,), (discovered, processed, scheduled_level, list_a, counter,
                                                 i32(current), i32(cap), i32(N)))
        K["af_finalize"]((1,), (1,), (counter, pending_count, pending_bits, out_dev, i32(current), i32(0), i32(cap)))
        prof.end("gather", tk)
        st["gather_calls"] += 1
        t = time.perf_counter(); cp.cuda.Stream.null.synchronize(); prof.add_host("sync", time.perf_counter() - t)
        t = time.perf_counter(); o = d2h(out_dev); prof.add_host("d2h", time.perf_counter() - t)
        n0, over, mism = int(o[0]), int(o[1]), int(o[2])
        check_guard()
        if mism:
            raise RuntimeError(f"gather/pending_count mismatch at level {current}: gathered {n0}, "
                               f"pending_count={_scalar_int(cp, pending_count[current])}")
        cur_list, nxt_list = list_a, list_b
        use_frozen = False
        if n0 == 0:
            pass  # nothing pending (cannot happen after select_next_level; frozen loop would also skip)
        elif over:
            st["visits_gather_overflow"] += 1
            use_frozen = True  # arrays untouched by the gather -> the frozen kernels can run the whole visit
        else:
            st["visits_list"] += 1
            n = n0
            cur_list, nxt_list = list_a, list_b
            while n > 0:
                rounds += 1
                total_activation_rounds += 1
                if total_activation_rounds > safety_limit:
                    raise RuntimeError(f"safety limit exceeded ({safety_limit}) while draining level {current}")
                grid_n = (n + block - 1) // block
                tk = prof.begin()
                K["af_round"]((grid_n,), (block,), (u, discovered, processed, scheduled_level, lo_f, hi_f,
                                                     pending_count, pending_bits, schedule_dir_count,
                                                     cur_list, i32(n), nxt_list, counter, i32(cap),
                                                     i32(current), i32(Wp), i32(Hp), i32(Dp)))
                prof.end("round", tk)
                tk = prof.begin()
                K["af_finalize"]((1,), (1,), (counter, pending_count, pending_bits, out_dev,
                                              i32(current), i32(1), i32(cap)))
                prof.end("finalize", tk)
                t = time.perf_counter(); cp.cuda.Stream.null.synchronize(); prof.add_host("sync", time.perf_counter() - t)
                t = time.perf_counter(); o = d2h(out_dev); prof.add_host("d2h", time.perf_counter() - t)
                total, over = int(o[0]), int(o[1])
                check_guard()
                st["rounds_list"] += 1
                st["max_frontier"] = max(st["max_frontier"], n)
                if trace:
                    frontier_sizes.append(n)
                if round_hook:
                    round_hook(dict(kind="list", round=total_activation_rounds, level=current, n=n,
                                    arrays=arrays))
                if over:
                    st["visits_mid_fallback"] += 1
                    use_frozen = True  # arrays == frozen post-round state; pending_count[current] == total
                    break
                cur_list, nxt_list = nxt_list, cur_list
                n = total
        if use_frozen:
            while True:
                t = time.perf_counter(); pend = _scalar_int(cp, pending_count[current]); prof.add_host("d2h", time.perf_counter() - t)
                if pend == 0:
                    break
                rounds += 1
                total_activation_rounds += 1
                if total_activation_rounds > safety_limit:
                    raise RuntimeError(f"safety limit exceeded ({safety_limit}) while draining level {current}")
                frozen_round(current)
                st["rounds_frozen"] += 1
                if round_hook:
                    round_hook(dict(kind="frozen", round=total_activation_rounds, level=current, n=None,
                                    arrays=arrays))

        closure_rounds_by_level[str(current)] = closure_rounds_by_level.get(str(current), 0) + rounds
        if trace:
            visit_depths.append(rounds)

        # ------------------------------ next level (frozen kernel, unchanged) -------------------------
        next_level[0] = -1
        tie_event[0] = 0
        has_next[0] = 0
        tk = prof.begin()
        K["select_next_level"]((1,), (1,), (pending_count, pending_bits, current, 0, next_level, tie_event, has_next))
        prof.end("select", tk)
        t = time.perf_counter(); cp.cuda.Stream.null.synchronize(); prof.add_host("sync", time.perf_counter() - t)
        t = time.perf_counter(); hn = _scalar_int(cp, has_next); prof.add_host("d2h", time.perf_counter() - t)
        if hn == 0:
            break
        nxt = _scalar_int(cp, next_level)
        if _scalar_int(cp, tie_event):
            tie_events += 1
        if nxt < current:
            backward_scheduler_transitions += 1
        level_transitions.append([int(current), int(nxt)])
        current = nxt

    prof.resolve()
    t_algo = time.perf_counter() - t_total0
    out = cp.asnumpy(u.reshape((Wp, Hp, Dp)))
    discovered_np = cp.asnumpy(discovered).astype(bool).reshape((Wp, Hp, Dp))
    processed_np = cp.asnumpy(processed).astype(bool).reshape((Wp, Hp, Dp))
    scheduled_np = cp.asnumpy(scheduled_level).reshape((Wp, Hp, Dp))
    pending_count_np = cp.asnumpy(pending_count)
    pending_bits_np = cp.asnumpy(pending_bits)
    dirs = cp.asnumpy(schedule_dir_count).astype(np.int64)
    frontier_np = cp.asnumpy(frontier)
    if int(pending_count_np.sum()) != 0 or bool(np.any(pending_bits_np)):
        raise RuntimeError("worklist terminated with non-empty pending state")
    if bool(np.any(frontier_np)):
        raise RuntimeError("frozen-fallback frontier array not clean at termination")

    meta: dict[str, Any] = {
        "implementation": IMPLEMENTATION,
        "implementation_version": "A1-v0",
        "base_frozen_sha256": FROZEN_SHA256,
        "bucket_count": N_LEVELS,
        "V_MIN": F.V_MIN,
        "V_MAX": F.V_MAX,
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
        "schedule_direction_counts": {"backward": int(dirs[0]), "same": int(dirs[1]), "forward": int(dirs[2])},
        "scheduler_transition_count": len(level_transitions),
        "backward_scheduler_transition_count": int(backward_scheduler_transitions),
        "equidistant_next_level_tie_events": int(tie_events),
        "total_activation_rounds": int(total_activation_rounds),
        "closure_rounds_by_level": closure_rounds_by_level,
        "level_transitions": level_transitions,
        "af": dict(cap=cap, N=N, stats=st, t_core_s=t_algo,
                   prof=dict(host_s=prof.host, gpu_ms=prof.gpu_ms, calls=prof.calls) if profile else None,
                   frontier_sizes=frontier_sizes if trace else None,
                   visit_depths=visit_depths if trace else None),
    }
    aux = {"discovered": discovered_np, "processed": processed_np, "scheduled_level": scheduled_np,
           "pending_count": pending_count_np, "pending_bits": pending_bits_np}
    return out, meta, aux
