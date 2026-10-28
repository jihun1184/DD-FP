// cuda_shim.h -- CPU "execution shim" for CUDA kernel *source text* (A1 gate, test tooling only).
//
// What it is   : lets the unmodified `__global__` functions of the frozen worklist and of the
//                active-frontier prototype be compiled as ordinary C++ and executed by real OS
//                threads with real atomics.  One launch = one parallel region; the caller joins
//                before the next launch (== kernel launch boundary on the null stream).
// What it is NOT: a GPU.  There is no warp lock-step, no GPU memory model, no nvcc front end.
//                Blocks run concurrently on OS threads; threads *inside* a block run one after
//                another in a seeded random order.  Evidence produced with it is tier "SHIM".
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline __attribute__((always_inline))

struct shim_dim3 { unsigned x, y, z; };
static thread_local shim_dim3 blockIdx  = {0, 0, 0};
static thread_local shim_dim3 threadIdx = {0, 0, 0};
static shim_dim3 blockDim = {1, 1, 1};
static shim_dim3 gridDim  = {1, 1, 1};

// ---- optional scheduling jitter: widens race windows around every atomic -------------------------
static double shim_jitter_p = 0.0;
static thread_local uint64_t shim_xs = 88172645463325252ull;
static inline void shim_jitter() {
    if (shim_jitter_p <= 0.0) return;
    shim_xs ^= shim_xs << 13; shim_xs ^= shim_xs >> 7; shim_xs ^= shim_xs << 17;
    if ((double)(shim_xs >> 11) / 9007199254740992.0 < shim_jitter_p) std::this_thread::yield();
}

// ---- CUDA atomics (sequentially consistent; CUDA's are relaxed, so this is only stronger) --------
static inline int atomicCAS(int* a, int cmp, int val) {
    shim_jitter();
    int e = cmp;
    __atomic_compare_exchange_n(a, &e, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    shim_jitter();
    return e;  // old value in both outcomes
}
static inline int atomicAdd(int* a, int v)  { shim_jitter(); int r = __atomic_fetch_add(a, v, __ATOMIC_SEQ_CST); shim_jitter(); return r; }
static inline int atomicSub(int* a, int v)  { shim_jitter(); int r = __atomic_fetch_sub(a, v, __ATOMIC_SEQ_CST); shim_jitter(); return r; }
static inline int atomicExch(int* a, int v) { shim_jitter(); int r = __atomic_exchange_n(a, v, __ATOMIC_SEQ_CST); shim_jitter(); return r; }
static inline unsigned atomicOr(unsigned* a, unsigned v)  { shim_jitter(); unsigned r = __atomic_fetch_or(a, v, __ATOMIC_SEQ_CST); shim_jitter(); return r; }
static inline unsigned atomicAnd(unsigned* a, unsigned v) { shim_jitter(); unsigned r = __atomic_fetch_and(a, v, __ATOMIC_SEQ_CST); shim_jitter(); return r; }

static inline int __float2int_rn(float f) { return (int)std::nearbyintf(f); }  // round-to-nearest-even

// ---- launcher ----------------------------------------------------------------------------------
static inline void shim_run(long grid, long block, unsigned long seed, const std::function<void()>& body) {
    blockDim.x = (unsigned)block; gridDim.x = (unsigned)grid;
    const char* e = std::getenv("SHIM_THREADS");
    int nth = e ? std::atoi(e) : 4;
    if (nth < 1) nth = 1;
    std::vector<long> perm((size_t)grid);
    std::iota(perm.begin(), perm.end(), 0L);
    { std::mt19937_64 g(seed * 2654435761ull + 17); std::shuffle(perm.begin(), perm.end(), g); }
    std::atomic<long> next(0);
    auto worker = [&](int wid) {
        std::mt19937_64 rng(seed * 1000003ull + (unsigned long)wid + 1);
        shim_xs = seed * 6364136223846793005ull + 1442695040888963407ull * (unsigned long)(wid + 1);
        std::vector<int> order((size_t)block);
        std::iota(order.begin(), order.end(), 0);
        for (;;) {
            long k = next.fetch_add(1);
            if (k >= grid) break;
            blockIdx.x = (unsigned)perm[(size_t)k];
            std::shuffle(order.begin(), order.end(), rng);
            for (int t : order) { threadIdx.x = (unsigned)t; body(); }
        }
    };
    if (nth == 1 || grid == 1) { worker(0); return; }
    std::vector<std::thread> th;
    for (int w = 0; w < nth; ++w) th.emplace_back(worker, w);
    for (auto& t : th) t.join();
}

extern "C" void shim_set_jitter(double p) { shim_jitter_p = p; }
