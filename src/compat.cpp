#include "compat.h"

namespace SplitEnum {
namespace {
bool edges_ok(const std::array<Point, 2>& e, const std::array<Point, 2>& f) {
  if (e == f) return false;
  const Point &p = e[0], &q = e[1], &r = f[0], &t = f[1];
  const long s1 = static_cast<long>(cross(p, q, r)) * cross(p, q, t);
  const long s2 = static_cast<long>(cross(r, t, p)) * cross(r, t, q);
  return !(s1 < 0 && s2 < 0);
}
}

bool splits_compatible(const SplitRec& a, const SplitRec& b) {
  if (!edges_ok(a.seg_a, b.seg_a)) return false;
  if (b.nsegs == 2 && !edges_ok(a.seg_a, b.seg_b)) return false;
  if (a.nsegs == 2) {
    if (!edges_ok(a.seg_b, b.seg_a)) return false;
    if (b.nsegs == 2 && !edges_ok(a.seg_b, b.seg_b)) return false;
  }
  return true;
}

Compat build_compat(const SplitTable& table) {
  Compat c;
  c.n = table.size();
  c.words = (c.n + 63) / 64;
  c.bits.assign(static_cast<size_t>(c.n) * c.words, 0);
  for (int i = 0; i < c.n; ++i) {
    uint64_t* ri = c.bits.data() + static_cast<size_t>(i) * c.words;
    for (int j = i + 1; j < c.n; ++j) {
      if (!splits_compatible(table.recs[i], table.recs[j])) continue;
      ri[j >> 6] |= uint64_t{1} << (j & 63);
      c.bits[static_cast<size_t>(j) * c.words + (i >> 6)] |= uint64_t{1}
                                                             << (i & 63);
    }
  }
  return c;
}
}
