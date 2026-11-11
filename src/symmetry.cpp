#include "symmetry.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>

namespace SplitEnum {
namespace {
Point apply_pt(int which, int delta, const Point& p) {
  const int x = p.first, y = p.second;
  switch (which) {
    case 1: return {y, x};
    case 2: return {delta - x - y, x};
    case 3: return {y, delta - x - y};
    case 4: return {x, delta - x - y};
    case 5: return {delta - x - y, y};
  }
  return p;
}

int image_cmp(const uint16_t* perm, const uint16_t* stack, int depth,
              uint16_t* scratch) {
  for (int i = 0; i < depth; ++i) {
    const uint16_t x = perm[stack[i]];
    int j = i;
    for (; j > 0 && scratch[j - 1] > x; --j) scratch[j] = scratch[j - 1];
    scratch[j] = x;
  }
  for (int i = 0; i < depth; ++i)
    if (scratch[i] != stack[i]) return scratch[i] < stack[i] ? -1 : 1;
  return 0;
}
}

Symmetry::Symmetry(const SplitTable& table) {
  const int delta = table.delta;
  const int n = table.size();

  const std::vector<Point> pts = lattice_points(delta);
  masks_ = table.affine;

  for (int which = 1; which <= 5; ++which) {
    SymElem e;
    e.split_perm.resize(n);
    bool closed = true;
    for (int i = 0; i < n && closed; ++i) {
      Split img = table.splits[i];
      for (Point& q : img) q = apply_pt(which, delta, q);
      if (img.back() < img.front()) std::reverse(img.begin(), img.end());
      const auto it =
          std::lower_bound(table.splits.begin(), table.splits.end(), img);
      if (it == table.splits.end() || *it != img) {
        closed = false;
        break;
      }
      e.split_perm[i] = static_cast<uint16_t>(it - table.splits.begin());
    }
    if (!closed) continue;

    std::vector<bool> seen(n, false);
    for (uint16_t v : e.split_perm) {
      if (seen[v])
        throw std::runtime_error(
            "symmetry self-check failed: split map is not a bijection");
      seen[v] = true;
    }
    e.inv_perm.resize(n);
    for (int i = 0; i < n; ++i) e.inv_perm[e.split_perm[i]] =
        static_cast<uint16_t>(i);

    e.nbytes = (table.npoints + 7) / 8;
    for (auto& lut : e.bit_lut) lut.fill(0);
    for (int i = 0; i < table.npoints; ++i) {
      const int j = flat_idx(delta, apply_pt(which, delta, pts[i]));
      const int byte = i >> 3;
      for (int v = 0; v < 256; ++v)
        if (v & (1 << (i & 7))) e.bit_lut[byte][v] |= SignMask{1} << j;
    }

    elems_.push_back(std::move(e));
  }

  std::set<std::vector<uint16_t>> members;
  std::vector<uint16_t> ident(n);
  std::iota(ident.begin(), ident.end(), 0);
  members.insert(ident);
  for (const SymElem& e : elems_) members.insert(e.split_perm);
  std::vector<uint16_t> comp(n);
  for (const auto& pa : members)
    for (const auto& pb : members) {
      for (int i = 0; i < n; ++i) comp[i] = pa[pb[i]];
      if (!members.count(comp))
        throw std::runtime_error(
            "symmetry self-check failed: kept elements are not closed under "
            "composition");
    }

  const auto is_mask = [&](SignMask d) {
    return std::find(masks_.begin(), masks_.end(), d) != masks_.end();
  };
  for (const SymElem& e : elems_) {
    if (!is_mask(e.apply_bits(table.harnack) ^ table.harnack))
      throw std::runtime_error(
          "symmetry self-check failed: Harnack signs are not affine-invariant "
          "under the group");
    for (int i = 0; i < n; ++i)
      if (!is_mask(e.apply_bits(table.recs[i].twist) ^
                   table.recs[e.split_perm[i]].twist))
        throw std::runtime_error(
            "symmetry self-check failed: twist of split " + std::to_string(i) +
            " is not equivariant up to an affine mask");
  }

  words_ = (n + 63) / 64;
  std::vector<uint16_t> minimg(n);
  for (int w = 0; w < n; ++w) {
    uint16_t m = static_cast<uint16_t>(w);
    for (const SymElem& e : elems_) m = std::min(m, e.split_perm[w]);
    minimg[w] = m;
  }
  allowed_.assign(static_cast<size_t>(n) * words_, 0);
  for (int u = 0; u < n; ++u) {
    uint64_t* row = allowed_.data() + static_cast<size_t>(u) * words_;
    for (int w = 0; w < n; ++w)
      if (minimg[w] >= u) row[w >> 6] |= uint64_t{1} << (w & 63);
  }
}

bool Symmetry::canonical_stack(const uint16_t* stack, int depth,
                               uint16_t* scratch, uint32_t& weight) const {
  int stab = 1;
  for (const SymElem& e : elems_) {
    const uint16_t* p = e.split_perm.data();
    if (depth) {
      uint16_t mn = p[stack[0]];
      for (int i = 1; i < depth; ++i) mn = std::min(mn, p[stack[i]]);
      if (mn < stack[0]) return false;
      if (mn > stack[0]) continue;
    }
    const int cmp = image_cmp(p, stack, depth, scratch);
    if (cmp < 0) return false;
    if (cmp == 0) ++stab;
  }
  weight = static_cast<uint32_t>(order() / stab);
  return true;
}

bool Symmetry::canonical_stack_active(const uint16_t* stack, int depth,
                                      uint16_t* scratch, unsigned active,
                                      uint32_t& weight) const {
  int stab = 1;
  for (; active; active &= active - 1) {
    const int cmp = image_cmp(elems_[__builtin_ctz(active)].split_perm.data(),
                              stack, depth, scratch);
    if (cmp < 0) return false;
    if (cmp == 0) ++stab;
  }
  weight = static_cast<uint32_t>(order() / stab);
  return true;
}
}
