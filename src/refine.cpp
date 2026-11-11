#include "refine.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace SplitEnum {
LeanRefine::LeanRefine(const SplitTable& table) {
  const int delta = table.delta;
  const int nverts = table.npoints;
  nedges_ = 3 * delta * (delta + 1) / 2;

  std::vector<std::pair<int, int>> pts(nverts);
  for (int i = 0; i < nverts; ++i) pts[i] = Isotopy::idx_to_point(delta, i);

  struct Cand {
    int len2;
    std::pair<int, int> e;
  };
  std::vector<Cand> cands;
  for (int i = 0; i < nverts; ++i)
    for (int j = i + 1; j < nverts; ++j) {
      const int dx = pts[j].first - pts[i].first;
      const int dy = pts[j].second - pts[i].second;
      if (std::gcd(std::abs(dx), std::abs(dy)) == 1)
        cands.push_back({dx * dx + dy * dy, {i, j}});
    }
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
    return std::tie(a.len2, a.e) < std::tie(b.len2, b.e);
  });

  pool_size_ = static_cast<int>(cands.size());
  if (pool_size_ > 64 * kWords)
    throw std::runtime_error("refine pool exceeds the fixed bitset width");
  for (int k = 0; k < pool_size_; ++k)
    valid_[k >> 6] |= uint64_t{1} << (k & 63);

  pool_.resize(2 * static_cast<size_t>(pool_size_));
  std::vector<int32_t> edge_id(static_cast<size_t>(nverts) * nverts, -1);
  for (int k = 0; k < pool_size_; ++k) {
    pool_[2 * k] = static_cast<uint8_t>(cands[k].e.first);
    pool_[2 * k + 1] = static_cast<uint8_t>(cands[k].e.second);
    edge_id[static_cast<size_t>(cands[k].e.first) * nverts + cands[k].e.second] = k;
  }

  const auto orient = [&](int i, int j, int k) {
    const int det =
        (pts[j].first - pts[i].first) * (pts[k].second - pts[i].second) -
        (pts[j].second - pts[i].second) * (pts[k].first - pts[i].first);
    return (det > 0) - (det < 0);
  };
  cross_.assign(static_cast<size_t>(pool_size_) * kWords, 0);
  for (int a = 0; a < pool_size_; ++a)
    for (int b = a + 1; b < pool_size_; ++b) {
      const int a0 = pool_[2 * a], a1 = pool_[2 * a + 1];
      const int b0 = pool_[2 * b], b1 = pool_[2 * b + 1];
      if (orient(a0, a1, b0) * orient(a0, a1, b1) < 0 &&
          orient(b0, b1, a0) * orient(b0, b1, a1) < 0) {
        cross_[static_cast<size_t>(a) * kWords + (b >> 6)] |= uint64_t{1} << (b & 63);
        cross_[static_cast<size_t>(b) * kWords + (a >> 6)] |= uint64_t{1} << (a & 63);
      }
    }

  const int n = table.size();
  forced_.assign(static_cast<size_t>(n) * kWords, 0);
  blocked_.assign(static_cast<size_t>(n) * kWords, 0);
  for (int s = 0; s < n; ++s) {
    const SplitRec& rec = table.recs[s];
    uint64_t* f = forced_.data() + static_cast<size_t>(s) * kWords;
    uint64_t* b = blocked_.data() + static_cast<size_t>(s) * kWords;
    for (int seg = 0; seg < rec.nsegs; ++seg) {
      const Isotopy::Edge e = seg == 0 ? rec.idx_a : rec.idx_b;
      const auto [lo, hi] = std::minmax(e.first, e.second);
      const int32_t id = edge_id[static_cast<size_t>(lo) * nverts + hi];
      if (id < 0)
        throw std::runtime_error("split edge is not a primitive pool edge");
      f[id >> 6] |= uint64_t{1} << (id & 63);
      const uint64_t* row = cross_.data() + static_cast<size_t>(id) * kWords;
      for (int w = 0; w < kWords; ++w) b[w] |= row[w];
    }
  }
}

bool LeanRefine::sweep(uint64_t* acc, const uint64_t* blk) const {
  uint64_t b[kWords];
  uint64_t clash = 0;
  int have = 0;
  for (int w = 0; w < kWords; ++w) {
    b[w] = blk[w];
    clash |= acc[w] & blk[w];
    have += __builtin_popcountll(acc[w]);
  }
  if (clash) return false;
  for (int w = 0; w < kWords && have < nedges_; ++w) {
    uint64_t a = acc[w];
    uint64_t avail = ~(a | b[w]) & valid_[w];
    while (avail) {
      const int bit = __builtin_ctzll(avail);
      const uint64_t* row =
          cross_.data() + static_cast<size_t>((w << 6) + bit) * kWords;
      a |= uint64_t{1} << bit;
      ++have;
      avail &= ~row[w] & (avail - 1);
      for (int x = 0; x < kWords; ++x) b[x] |= row[x];
    }
    acc[w] = a;
  }
  return have == nedges_;
}

bool LeanRefine::refine(const uint16_t* ids, int n, uint64_t* acc) const {
  uint64_t blk[kWords] = {};
  std::fill(acc, acc + kWords, uint64_t{0});
  for (int i = 0; i < n; ++i) {
    const uint64_t* f = forced_row(ids[i]);
    const uint64_t* b = blocked_row(ids[i]);
    for (int w = 0; w < kWords; ++w) {
      acc[w] |= f[w];
      blk[w] |= b[w];
    }
  }
  return sweep(acc, blk);
}

void LeanRefine::edges(const uint64_t* acc, uint8_t* out) const {
  for (int w = 0; w < kWords; ++w)
    for (uint64_t m = acc[w]; m; m &= m - 1) {
      const int k = (w << 6) + __builtin_ctzll(m);
      *out++ = pool_[2 * k];
      *out++ = pool_[2 * k + 1];
    }
}

bool LeanRefine::matches_library(const SplitTable& table, const uint16_t* ids,
                                 int n) const {
  std::vector<Isotopy::Edge> forced;
  for (int i = 0; i < n; ++i) {
    const SplitRec& rec = table.recs[ids[i]];
    forced.push_back(rec.idx_a);
    if (rec.nsegs == 2) forced.push_back(rec.idx_b);
  }
  const auto tris = Isotopy::refine(table.delta, forced);
  uint64_t acc[kWords];
  const bool ok = refine(ids, n, acc);
  if (!tris || !ok) return !tris && !ok;

  std::vector<std::pair<int, int>> want;
  for (const Isotopy::Edge& e : Isotopy::triangles_to_edges(*tris))
    want.push_back(std::minmax(e.first, e.second));
  std::sort(want.begin(), want.end());
  want.erase(std::unique(want.begin(), want.end()), want.end());

  std::vector<uint8_t> bytes(2 * static_cast<size_t>(nedges_));
  edges(acc, bytes.data());
  std::vector<std::pair<int, int>> got;
  for (int e = 0; e < nedges_; ++e)
    got.emplace_back(bytes[2 * e], bytes[2 * e + 1]);
  std::sort(got.begin(), got.end());
  return got == want;
}
}
