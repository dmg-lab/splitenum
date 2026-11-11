#pragma once

#include <cstdint>
#include <vector>

#include "splits.h"

namespace SplitEnum {
class LeanRefine {
 public:
  static constexpr int kWords = 16;

  explicit LeanRefine(const SplitTable& table);

  int nedges() const { return nedges_; }
  int pool_size() const { return pool_size_; }

  const uint64_t* forced_row(int split) const {
    return forced_.data() + static_cast<size_t>(split) * kWords;
  }
  const uint64_t* blocked_row(int split) const {
    return blocked_.data() + static_cast<size_t>(split) * kWords;
  }
  int nsplits() const { return static_cast<int>(forced_.size() / kWords); }
  const uint64_t* forced_rows() const { return forced_.data(); }
  const uint64_t* blocked_rows() const { return blocked_.data(); }
  const uint64_t* cross_rows() const { return cross_.data(); }
  const uint8_t* pool() const { return pool_.data(); }
  const uint64_t* valid() const { return valid_; }

  bool sweep(uint64_t* acc, const uint64_t* blk) const;
  bool refine(const uint16_t* ids, int n, uint64_t* acc) const;
  void edges(const uint64_t* acc, uint8_t* out) const;
  bool matches_library(const SplitTable& table, const uint16_t* ids,
                       int n) const;

 private:
  int nedges_ = 0;
  int pool_size_ = 0;
  uint64_t valid_[kWords] = {};
  std::vector<uint8_t> pool_;
  std::vector<uint64_t> cross_;
  std::vector<uint64_t> forced_;
  std::vector<uint64_t> blocked_;
};
}
