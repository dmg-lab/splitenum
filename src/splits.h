#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "haas.h"
#include "isotopy_graph.h"

namespace SplitEnum {
using Isotopy::Edge;
using Isotopy::Point;
using Isotopy::Split;

using SignMask = uint64_t;
constexpr int MAX_SUPPORTED_DELTA = 9;

enum class Family {
  All,
  WithZero,
  WithoutZero
};

struct SplitRec {
  std::array<Point, 2> seg_a;
  std::array<Point, 2> seg_b;
  Edge idx_a;
  Edge idx_b;
  SignMask twist;
  uint8_t nsegs;
  std::array<uint8_t, 3> edge_dist{};
};

using Profile = uint32_t;
inline constexpr Profile kEmptyProfile = 0xffffff;

struct SplitTable {
  int delta = 0;
  int npoints = 0;
  SignMask harnack = 0;
  std::array<SignMask, 8> affine{};
  std::vector<Split> splits;
  std::vector<SplitRec> recs;

  int size() const { return static_cast<int>(splits.size()); }
};

SplitTable build_splits(int delta, Family family, bool drop_same_side,
                        bool drop_trivial, bool drop_collinear);

std::vector<Split> raw_splits(int delta, Family family, bool drop_same_side,
                              bool drop_collinear);

Profile profile_of(const SplitTable& table, const uint16_t* ids, int n);
int support_degree(int delta, Profile p);
std::string profile_text(Profile p);

std::vector<bool> unpack_signs(SignMask sig, int npoints);

inline int flat_idx(int delta, const Point& p) {
  return p.second * (delta + 1) - p.second * (p.second - 1) / 2 + p.first;
}

inline int cross(const Point& o, const Point& a, const Point& b) {
  return (a.first - o.first) * (b.second - o.second) -
         (a.second - o.second) * (b.first - o.first);
}

std::vector<Point> lattice_points(int delta);

template <typename Pred>
SignMask point_mask(int delta, Pred&& pred) {
  SignMask m = 0;
  for (int y = 0; y <= delta; ++y)
    for (int x = 0; x <= delta - y; ++x)
      if (pred(x, y)) m |= SignMask{1} << flat_idx(delta, {x, y});
  return m;
}

const char* family_name(Family f);
bool parse_family(const std::string& s, Family& out);
}
