#pragma once

#include <climits>
#include <cstddef>
#include <cstdint>

#include "isotopy_gpu.cuh"

namespace SplitEnum {
inline constexpr int kRefineWords = 16;
inline constexpr uint32_t kRefineFailed = 1u << 16;

struct RefineTables {
  int32_t nedges;
  int32_t pool_size;
  int32_t nsplits;
  uint64_t valid[kRefineWords];
  const uint64_t* forced;
  const uint64_t* blocked;
  const uint64_t* cross;
  const uint8_t* pool;
};

struct RefineScratch {
  uint64_t acc[kRefineWords];
  uint64_t blk[kRefineWords];
};

ISO_HD inline int ctz64(uint64_t x) {
#ifdef __CUDA_ARCH__
  return __ffsll(static_cast<long long>(x)) - 1;
#else
  return __builtin_ctzll(x);
#endif
}

ISO_HD inline int popc64(uint64_t x) {
#ifdef __CUDA_ARCH__
  return __popcll(x);
#else
  return __builtin_popcountll(x);
#endif
}

template <int LANES>
ISO_HD inline void owned_words(IsotopyGPU::Warp<LANES> w, int& lo, int& hi) {
  constexpr int per = (kRefineWords + LANES - 1) / LANES;
  lo = w.lane() * per;
  if (lo > kRefineWords) lo = kRefineWords;
  hi = lo + per < kRefineWords ? lo + per : kRefineWords;
}

template <int LANES>
ISO_HD bool sweep_warp(const RefineTables& t, const uint16_t* ids, int n,
                       RefineScratch& s, IsotopyGPU::Warp<LANES> w) {
  int lo, hi;
  owned_words(w, lo, hi);
  bool clash = false;
  for (int x = lo; x < hi; ++x) {
    uint64_t a = 0, b = 0;
    for (int i = 0; i < n; ++i) {
      const size_t row = static_cast<size_t>(ids[i]) * kRefineWords + x;
      a |= t.forced[row];
      b |= t.blocked[row];
    }
    s.acc[x] = a;
    s.blk[x] = b;
    clash = clash || (a & b) != 0;
  }
  if (w.any(clash)) return false;
  for (;;) {
    int k = INT_MAX;
    for (int x = lo; x < hi && k == INT_MAX; ++x) {
      const uint64_t avail = ~(s.acc[x] | s.blk[x]) & t.valid[x];
      if (avail) k = (x << 6) + ctz64(avail);
    }
    k = w.reduce_min(k);
    if (k == INT_MAX) break;
    const uint64_t* row = t.cross + static_cast<size_t>(k) * kRefineWords;
    for (int x = lo; x < hi; ++x) s.blk[x] |= row[x];
    const int word = k >> 6;
    if (word >= lo && word < hi) s.acc[word] |= uint64_t{1} << (k & 63);
  }
  int have = 0;
  for (int x = lo; x < hi; ++x) have += popc64(s.acc[x]);
  return w.reduce_add(have) == t.nedges;
}

template <int LANES>
ISO_HD void expand_edges(const RefineTables& t, const RefineScratch& s,
                         uint8_t* out, IsotopyGPU::Warp<LANES> w) {
  int lo, hi;
  owned_words(w, lo, hi);
  int mine = 0;
  for (int x = lo; x < hi; ++x) mine += popc64(s.acc[x]);
  int total = 0;
  int at = w.exclusive_scan(mine, &total);
  for (int x = lo; x < hi; ++x)
    for (uint64_t m = s.acc[x]; m; m &= m - 1) {
      const int k = (x << 6) + ctz64(m);
      out[2 * at] = t.pool[2 * k];
      out[2 * at + 1] = t.pool[2 * k + 1];
      ++at;
    }
  w.sync();
}

template <int LANES>
ISO_HD void refine_classify_one(const IsotopyGPU::Tables& tables,
                                const RefineTables& rt, const uint16_t* ids,
                                int n, const uint64_t* signs,
                                RefineScratch& rs, uint8_t* edges,
                                IsotopyGPU::Scratch& sh,
                                IsotopyGPU::Warp<LANES> w,
                                IsotopyGPU::Result& out) {
  if (!sweep_warp(rt, ids, n, rs, w)) {
    out = IsotopyGPU::Result{};
    out.flags = kRefineFailed;
    out.type_id = IsotopyGPU::kTypeFallback;
    return;
  }
  expand_edges(rt, rs, edges, w);
  IsotopyGPU::classify_one(tables, edges, signs, sh, w, out);
}
}
