"""
generic_worklists_v2.py -- (v2 of comparison/generic_worklists.py; v1 untouched) minimal GENERIC GPU worklists for front propagation (comparison baselines only).

NOT part of AF, frozen or production.  Nothing here imports af_worklist's kernels; only the frozen kernel text
(``snap_level`` and ``select_next_level``, verbatim) is reused so that "the FP contract" is the same object.

Purpose (CMPB novelty gate): decide which parts of AF are FP-specific and necessary, by running *textbook* GPU
worklist designs against the same contract (first-discovery clamp, 256 level buckets, nearest-occupied-level
selection that may move backwards, lower tie, round boundary, final state arrays).

Variants
  pile  / nearest-lower   : IWPP / Near-Far style.  Every discovered cell whose level differs from the current level is
                            appended ONCE to a global pile; at each level visit the pile is split by a scan+compaction
                            into (cells of the selected level, remaining pile).  Same-level discoveries go to the next
                            round's list (round boundary kept).  No full-grid scan after start-up.  No fallback:
                            capacity overflow is an error (a generic implementation would size the arrays for the
                            worst case).
  pile  / nearest-higher  : same, tie policy "higher".
  pile  / lowest-first    : generic priority-queue order (always the lowest occupied level).
  pile  / monotone-wrap   : ADDS / delta-stepping style monotone sweep (next occupied level >= current, wrap-around).
  pile  / highest-first   : mirror of lowest-first.
  bfs                     : level-agnostic round-synchronous frontier (Harish-Narayanan / Merrill style): every
                            discovered cell is expanded in discovery order; the clamp uses the level of the cell that
                            discovered it.  No level buckets at all.
Only the first two are intended to be FP-correct.  The others are controls that show which design choices the FP
contract rules out.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import af_worklist as AF  # noqa: E402  (only for load_frozen / FROZEN_SHA256)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g5_rules import GenericInvariantError, GenericResourceError  # noqa: E402

# v2 changes vs v1: (1) typed errors (capacity -> GenericResourceError, invariant/safety -> GenericInvariantError) so that the G5 runner can tell a
# resource limit from an implementation defect; (2) the unused N-sized `frontier` array is no longer allocated; (3) allocation and occupancy are
# reported in meta["gw"]["alloc"]/["occupancy"].  Kernels (GW_CODE) and the algorithm are byte-for-byte those of v1.

IMPLEMENTATION = "GENERIC_PILE_SPLIT_COMPARISON_V2"

POLICIES = {"nearest-lower": ("frozen", 0), "nearest-higher": ("frozen", 1),
            "lowest-first": ("gw", 1), "monotone-wrap": ("gw", 2), "highest-first": ("gw", 3)}

GW_CODE = r'''
extern "C" {

// split the pile: cells scheduled at `level` -> list_out, all others -> pile_out (stream compaction, order irrelevant)
__global__ void gw_split(
    const int* pile_in, int n_in, const int* scheduled_level, int level,
    int* list_out, int* ctr, int* pile_out, int list_cap, int pile_cap
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_in) return;
    int idx = pile_in[i];
    if (scheduled_level[idx] == level) {
        int pos = atomicAdd(&ctr[0], 1);
        if (pos < list_cap) list_out[pos] = idx;
    } else {
        int pos = atomicAdd(&ctr[1], 1);
        if (pos < pile_cap) pile_out[pos] = idx;
    }
}

// one closure round on a compact list: same bookkeeping as the frozen activate+expand, textbook queue discipline
__global__ void gw_round(
    float* u, int* discovered, int* processed, int* scheduled_level,
    const float* U_lo, const float* U_hi,
    int* pending_count, unsigned int* pending_bits, int* schedule_dir_count,
    const int* list_cur, int n_cur, int* list_next, int* ctr, int* pile,
    int list_cap, int pile_cap, int current_level, int Wp, int Hp, int Dp
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cur) return;
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
        if (nx < 0 || nx >= Wp || ny < 0 || ny >= Hp || nz < 0 || nz >= Dp) continue;
        int nidx = (nx * Hp + ny) * Dp + nz;
        if (atomicCAS(&discovered[nidx], 0, 1) != 0) continue;
        int target = snap_level(current_level, U_lo[nidx], U_hi[nidx]);
        scheduled_level[nidx] = target;
        if (target == current_level) {
            int pos = atomicAdd(&ctr[0], 1);
            if (pos < list_cap) list_next[pos] = nidx;
            atomicAdd(&schedule_dir_count[1], 1);
        } else {
            atomicAdd(&pending_count[target], 1);
            atomicOr(&pending_bits[target >> 5], 1u << (target & 31));
            int pos = atomicAdd(&ctr[1], 1);
            if (pos < pile_cap) pile[pos] = nidx;
            if (target < current_level) atomicAdd(&schedule_dir_count[0], 1);
            else atomicAdd(&schedule_dir_count[2], 1);
        }
    }
}

// end of round: publish the same-level count (unclamped) as the new pending count of `level`
__global__ void gw_finalize(
    int* ctr, int* pending_count, unsigned int* pending_bits, int* out, int level, int list_cap, int pile_cap
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int total = ctr[0];
    ctr[0] = 0;
    out[0] = total;
    out[1] = (total > list_cap || ctr[1] > pile_cap) ? 1 : 0;
    out[2] = ctr[1];
    pending_count[level] = total;
    unsigned int bit = 1u << (level & 31);
    if (total > 0) atomicOr(&pending_bits[level >> 5], bit);
    else atomicAnd(&pending_bits[level >> 5], ~bit);
}

// non-FP level selection rules (controls)
__global__ void gw_select_policy(
    const int* pending_count, const unsigned int* pending_bits, int current_level, int policy,
    int* next_level, int* tie_event, int* has_next
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int best = -1;
    for (int level = 0; level < 256; ++level) {
        unsigned int mask = 1u << (level & 31);
        if ((pending_bits[level >> 5] & mask) == 0u) continue;
        if (pending_count[level] <= 0) continue;
        if (policy == 1) { if (best < 0) best = level; }                       // lowest occupied
        else if (policy == 3) { best = level; }                                // highest occupied
        else if (policy == 2) { if (level >= current_level && best < 0) best = level; }  // monotone sweep
    }
    if (policy == 2 && best < 0) {                                              // wrap around
        for (int level = 0; level < 256; ++level) {
            unsigned int mask = 1u << (level & 31);
            if ((pending_bits[level >> 5] & mask) == 0u) continue;
            if (pending_count[level] <= 0) continue;
            best = level; break;
        }
    }
    *tie_event = 0;
    if (best < 0) { *next_level = -1; *has_next = 0; } else { *next_level = best; *has_next = 1; }
}

// level-agnostic frontier round: clamp against the level of the discovering cell
__global__ void gw_bfs_round(
    float* u, int* discovered, const float* U_lo, const float* U_hi, int* schedule_dir_count,
    const int* list_cur, int n_cur, int* list_next, int* counter, int cap, int Wp, int Hp, int Dp
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cur) return;
    int idx = list_cur[i];
    int lv = (int)u[idx];
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
        if (nx < 0 || nx >= Wp || ny < 0 || ny >= Hp || nz < 0 || nz >= Dp) continue;
        int nidx = (nx * Hp + ny) * Dp + nz;
        if (atomicCAS(&discovered[nidx], 0, 1) != 0) continue;
        int target = snap_level(lv, U_lo[nidx], U_hi[nidx]);
        u[nidx] = (float)target;
        int pos = atomicAdd(counter, 1);
        if (pos < cap) list_next[pos] = nidx;
        if (target < lv) atomicAdd(&schedule_dir_count[0], 1);
        else if (target == lv) atomicAdd(&schedule_dir_count[1], 1);
        else atomicAdd(&schedule_dir_count[2], 1);
    }
}

} // extern "C"
'''


def _scalar_int(cp, a) -> int:
    return int(cp.asnumpy(a).reshape(-1)[0])


def _setup(cp, U_lo_pad, U_hi_pad, l_inf, frozen):
    F = frozen or AF.load_frozen()
    lo_dev = cp.ascontiguousarray(cp.asarray(U_lo_pad, dtype=cp.float32))
    hi_dev = cp.ascontiguousarray(cp.asarray(U_hi_pad, dtype=cp.float32))
    lo_host, hi_host, l0 = F._validate_host_intervals(cp.asnumpy(lo_dev).reshape(lo_dev.shape),
                                                      cp.asnumpy(hi_dev).reshape(hi_dev.shape), l_inf)
    Wp, Hp, Dp = (int(v) for v in lo_host.shape)
    N = Wp * Hp * Dp
    if N >= 2**31 - 1:
        raise ValueError("grid too large for int32 indexing")
    lo_f = cp.ascontiguousarray(lo_dev.ravel(), dtype=cp.float32)
    hi_f = cp.ascontiguousarray(hi_dev.ravel(), dtype=cp.float32)
    return F, lo_f, hi_f, l0, Wp, Hp, Dp, N


def generic_pile_worklist(cp, U_lo_pad, U_hi_pad, l_inf: float, policy: str = "nearest-lower", block_size: int = 256,
                          list_cap: int | None = None, pile_cap: int | None = None, *, frozen=None, trace: bool = False):
    """Pile-and-split generic worklist.  Returns (out, meta, aux) like boutry_bucket_gpu."""
    if policy not in POLICIES:
        raise ValueError(policy)
    F, lo_f, hi_f, l0, Wp, Hp, Dp, N = _setup(cp, U_lo_pad, U_hi_pad, l_inf, frozen)
    sel_kind, sel_arg = POLICIES[policy]
    N_LEVELS = F.N_LEVELS
    list_cap = int(min(N, AF.DEFAULT_CAP if list_cap is None else list_cap))
    pile_cap = int(N if pile_cap is None else min(N, pile_cap))  # every cell enters the pile at most once => N always suffices

    u = cp.full(N, cp.nan, dtype=cp.float32)
    discovered = cp.zeros(N, dtype=cp.int32)
    processed = cp.zeros(N, dtype=cp.int32)
    scheduled_level = cp.full(N, -1, dtype=cp.int32)
    pending_count = cp.zeros(N_LEVELS, dtype=cp.int32)
    pending_bits = cp.zeros((N_LEVELS + 31) // 32, dtype=cp.uint32)
    schedule_dir_count = cp.zeros(3, dtype=cp.int32)
    next_level = cp.full(1, -1, dtype=cp.int32)
    tie_event = cp.zeros(1, dtype=cp.int32)
    has_next = cp.zeros(1, dtype=cp.int32)
    list_a = cp.zeros(list_cap, dtype=cp.int32)
    list_b = cp.zeros(list_cap, dtype=cp.int32)
    pile_a = cp.zeros(pile_cap, dtype=cp.int32)
    pile_b = cp.zeros(pile_cap, dtype=cp.int32)
    ctr = cp.zeros(4, dtype=cp.int32)
    out_dev = cp.zeros(4, dtype=cp.int32)
    alloc = {k: int(a.nbytes) for k, a in (("u", u), ("discovered", discovered), ("processed", processed), ("scheduled_level", scheduled_level),
                                            ("pending_count", pending_count), ("pending_bits", pending_bits), ("list_a", list_a), ("list_b", list_b),
                                            ("pile_a", pile_a), ("pile_b", pile_b), ("ctr", ctr), ("out_dev", out_dev))}
    alloc["inputs_lo_hi_flat"] = int(lo_f.nbytes + hi_f.nbytes)

    discovered[0] = 1
    scheduled_level[0] = l0
    pending_count[l0] = 1
    pending_bits[l0 >> 5] = cp.uint32(1 << (l0 & 31))
    pile_a[0] = 0
    n_pile = 1

    module = cp.RawModule(code=F._RAW_KERNEL_CODE + "\n" + GW_CODE, options=("-std=c++17",))
    K = {n: module.get_function(n) for n in ("select_next_level", "gw_split", "gw_round", "gw_finalize", "gw_select_policy")}
    block = int(block_size)
    i32 = np.int32
    safety_limit = max(10000, N + 10 * N_LEVELS)
    level_transitions: list[list[int]] = []
    closure_rounds_by_level: dict[str, int] = {}
    total_activation_rounds = 0
    backward = 0
    tie_events = 0
    st = dict(visits=0, rounds=0, split_launches=0, split_elements=0, max_pile=1, max_frontier=0, max_list_next=0)
    frontier_sizes: list[int] = []
    pile_sizes: list[int] = []

    pile_cur, pile_alt = pile_a, pile_b
    current = l0
    while True:
        rounds = 0
        st["visits"] += 1
        # ---- visit start: split the pile (cost ~ pile size, never ~ N) -------------------------------------------
        ctr[0] = 0
        ctr[1] = 0
        if n_pile > 0:
            K["gw_split"](((n_pile + block - 1) // block,), (block,),
                          (pile_cur, i32(n_pile), scheduled_level, i32(current), list_a, ctr, pile_alt, i32(list_cap), i32(pile_cap)))
            st["split_launches"] += 1
            st["split_elements"] += n_pile
        cp.cuda.Stream.null.synchronize()
        c = cp.asnumpy(ctr)
        n0, n_pile_new = int(c[0]), int(c[1])
        if n0 > list_cap or n_pile_new > pile_cap:
            raise GenericResourceError(f"generic worklist capacity exceeded (list {n0}/{list_cap}, pile {n_pile_new}/{pile_cap}); no fallback")
        pile_cur, pile_alt = pile_alt, pile_cur
        n_pile = n_pile_new
        pile_sizes.append(n_pile)
        st["max_pile"] = max(st["max_pile"], n_pile)
        ctr[0] = 0
        if n0 != _scalar_int(cp, pending_count[current]):
            raise GenericInvariantError(f"split/pending_count mismatch at level {current}: split {n0}, pending_count={_scalar_int(cp, pending_count[current])}")
        cur_list, nxt_list = list_a, list_b
        n = n0
        while n > 0:
            rounds += 1
            total_activation_rounds += 1
            if total_activation_rounds > safety_limit:
                raise GenericInvariantError(f"safety limit exceeded ({safety_limit}) while draining level {current}")
            K["gw_round"](((n + block - 1) // block,), (block,),
                          (u, discovered, processed, scheduled_level, lo_f, hi_f, pending_count, pending_bits, schedule_dir_count,
                           cur_list, i32(n), nxt_list, ctr, pile_cur, i32(list_cap), i32(pile_cap), i32(current), i32(Wp), i32(Hp), i32(Dp)))
            K["gw_finalize"]((1,), (1,), (ctr, pending_count, pending_bits, out_dev, i32(current), i32(list_cap), i32(pile_cap)))
            cp.cuda.Stream.null.synchronize()
            o = cp.asnumpy(out_dev)
            total, over, n_pile = int(o[0]), int(o[1]), int(o[2])
            st["rounds"] += 1
            st["max_frontier"] = max(st["max_frontier"], n)
            st["max_list_next"] = max(st["max_list_next"], total)
            st["max_pile"] = max(st["max_pile"], n_pile)
            if trace:
                frontier_sizes.append(n)
            if over:
                raise GenericResourceError("generic worklist capacity exceeded inside a round; no fallback")
            cur_list, nxt_list = nxt_list, cur_list
            n = total
        closure_rounds_by_level[str(current)] = closure_rounds_by_level.get(str(current), 0) + rounds

        # ---- next level ------------------------------------------------------------------------------------------
        next_level[0] = -1
        tie_event[0] = 0
        has_next[0] = 0
        if sel_kind == "frozen":
            K["select_next_level"]((1,), (1,), (pending_count, pending_bits, current, i32(sel_arg), next_level, tie_event, has_next))
        else:
            K["gw_select_policy"]((1,), (1,), (pending_count, pending_bits, i32(current), i32(sel_arg), next_level, tie_event, has_next))
        cp.cuda.Stream.null.synchronize()
        if _scalar_int(cp, has_next) == 0:
            break
        nxt = _scalar_int(cp, next_level)
        if _scalar_int(cp, tie_event):
            tie_events += 1
        if nxt < current:
            backward += 1
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
        raise GenericInvariantError("worklist terminated with non-empty pending state")
    meta: dict[str, Any] = {
        "implementation": IMPLEMENTATION, "policy": policy, "tie_policy": "higher" if policy == "nearest-higher" else "lower",
        "nan_count": int(np.isnan(out).sum()), "discovered_count": int(discovered_np.sum()), "processed_count": int(processed_np.sum()),
        "pending_total_final": int(pending_count_np.sum()), "pending_bitset_nonzero_words_final": int(np.count_nonzero(pending_bits_np)),
        "schedule_direction_counts": {"backward": int(dirs[0]), "same": int(dirs[1]), "forward": int(dirs[2])},
        "scheduler_transition_count": len(level_transitions), "backward_scheduler_transition_count": int(backward),
        "equidistant_next_level_tie_events": int(tie_events), "total_activation_rounds": int(total_activation_rounds),
        "closure_rounds_by_level": closure_rounds_by_level, "level_transitions": level_transitions,
        "gw": dict(N=N, list_cap=list_cap, pile_cap=pile_cap, stats=st,
                   alloc=dict(bytes=alloc, total_mib=sum(alloc.values()) / 2**20, state_and_work_mib=(sum(alloc.values()) - alloc["inputs_lo_hi_flat"]) / 2**20),
                   occupancy=dict(max_pile=st["max_pile"], pile_cap=pile_cap, max_pile_over_cap=st["max_pile"] / pile_cap,
                                  max_frontier=st["max_frontier"], max_list_next=st["max_list_next"], list_cap=list_cap,
                                  max_list_over_cap=max(st["max_frontier"], st["max_list_next"]) / list_cap), pile_sizes=pile_sizes if trace else None,
                   frontier_sizes=frontier_sizes if trace else None,
                   full_grid_scan_elements_avoided=int(st["visits"]) * N - int(st["split_elements"])),
    }
    aux = {"discovered": discovered_np, "processed": processed_np, "scheduled_level": scheduled_np,
           "pending_count": pending_count_np, "pending_bits": pending_bits_np}
    return out, meta, aux


def generic_level_agnostic_frontier(cp, U_lo_pad, U_hi_pad, l_inf: float, block_size: int = 256, cap: int | None = None, *, frozen=None):
    """Level-agnostic round-synchronous frontier (control).  Output u only; there is no level-bucket state."""
    F, lo_f, hi_f, l0, Wp, Hp, Dp, N = _setup(cp, U_lo_pad, U_hi_pad, l_inf, frozen)
    cap = int(min(N, AF.DEFAULT_CAP if cap is None else cap))
    u = cp.full(N, cp.nan, dtype=cp.float32)
    discovered = cp.zeros(N, dtype=cp.int32)
    dirs_dev = cp.zeros(3, dtype=cp.int32)
    list_a = cp.zeros(cap, dtype=cp.int32)
    list_b = cp.zeros(cap, dtype=cp.int32)
    counter = cp.zeros(1, dtype=cp.int32)
    discovered[0] = 1
    u[0] = np.float32(l0)
    list_a[0] = 0
    module = cp.RawModule(code=F._RAW_KERNEL_CODE + "\n" + GW_CODE, options=("-std=c++17",))
    kern = module.get_function("gw_bfs_round")
    block = int(block_size)
    i32 = np.int32
    cur, nxt = list_a, list_b
    n, rounds = 1, 0
    safety_limit = max(10000, N + 10 * F.N_LEVELS)
    while n > 0:
        rounds += 1
        if rounds > safety_limit:
            raise RuntimeError("safety limit exceeded")
        counter[0] = 0
        kern(((n + block - 1) // block,), (block,), (u, discovered, lo_f, hi_f, dirs_dev, cur, i32(n), nxt, counter, i32(cap), i32(Wp), i32(Hp), i32(Dp)))
        cp.cuda.Stream.null.synchronize()
        total = _scalar_int(cp, counter)
        if total > cap:
            raise RuntimeError("frontier capacity exceeded; no fallback")
        cur, nxt = nxt, cur
        n = total
    out = cp.asnumpy(u.reshape((Wp, Hp, Dp)))
    dirs = cp.asnumpy(dirs_dev).astype(np.int64)
    meta = {"implementation": IMPLEMENTATION, "policy": "level-agnostic-frontier", "nan_count": int(np.isnan(out).sum()),
            "discovered_count": int(cp.asnumpy(discovered).sum()), "total_activation_rounds": int(rounds),
            "schedule_direction_counts": {"backward": int(dirs[0]), "same": int(dirs[1]), "forward": int(dirs[2])}}
    return out, meta, {}
