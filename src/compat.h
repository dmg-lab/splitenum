#pragma once

#include <cstdint>
#include <vector>

#include "splits.h"

namespace SplitEnum {
struct Compat {
  int n = 0;
  int words = 0;
  std::vector<uint64_t> bits;

  const uint64_t* row(int i) const {
    return bits.data() + static_cast<size_t>(i) * words;
  }
  bool get(int i, int j) const { return (row(i)[j >> 6] >> (j & 63)) & 1u; }
};

Compat build_compat(const SplitTable& table);

bool splits_compatible(const SplitRec& a, const SplitRec& b);

template <bool Masked>
inline bool and_above_impl(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                           const uint64_t* m, int words, int v) {
  const int lo = (v + 1) >> 6;
  const int off = (v + 1) & 63;
  const auto word = [&](int w) {
    return Masked ? a[w] & b[w] & m[w] : a[w] & b[w];
  };
  for (int w = 0; w < lo && w < words; ++w) dst[w] = 0;
  if (lo >= words) return false;
  dst[lo] = word(lo) & (off ? ~uint64_t{0} << off : ~uint64_t{0});
  uint64_t any = dst[lo];
  for (int w = lo + 1; w < words; ++w) {
    dst[w] = word(w);
    any |= dst[w];
  }
  return any != 0;
}

inline bool and_above(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                      int words, int v) {
  return and_above_impl<false>(dst, a, b, nullptr, words, v);
}

inline bool and_above_masked(uint64_t* dst, const uint64_t* a,
                             const uint64_t* b, const uint64_t* m, int words,
                             int v) {
  return and_above_impl<true>(dst, a, b, m, words, v);
}
}
