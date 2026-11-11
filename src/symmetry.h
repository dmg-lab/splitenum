#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "splits.h"

namespace SplitEnum {
struct SymElem {
  std::vector<uint16_t> split_perm;
  std::vector<uint16_t> inv_perm;
  std::array<std::array<SignMask, 256>, 8> bit_lut;
  int nbytes = 8;

  SignMask apply_bits(SignMask s) const {
    SignMask r = 0;
    for (int b = 0; b < nbytes; ++b) r |= bit_lut[b][(s >> (8 * b)) & 0xff];
    return r;
  }
};

class Symmetry {
 public:
  explicit Symmetry(const SplitTable& table);

  int order() const { return static_cast<int>(elems_.size()) + 1; }

  const std::vector<SymElem>& elems() const { return elems_; }

  bool canonical_stack(const uint16_t* stack, int depth, uint16_t* scratch,
                       uint32_t& weight) const;

  int words() const { return words_; }

  const uint64_t* allowed_row(int u) const {
    return allowed_.data() + static_cast<size_t>(u) * words_;
  }

  bool canonical_stack_active(const uint16_t* stack, int depth,
                              uint16_t* scratch, unsigned active,
                              uint32_t& weight) const;

 private:
  std::vector<SymElem> elems_;
  std::array<SignMask, 8> masks_{};
  int words_ = 0;
  std::vector<uint64_t> allowed_;
};
}
