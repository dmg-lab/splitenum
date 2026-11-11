#include "splits.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>

#include "compat.h"

namespace SplitEnum {
namespace {
bool on_segment(const Point& a, const Point& b, const Point& q) {
  return std::min(a.first, b.first) <= q.first &&
         q.first <= std::max(a.first, b.first) &&
         std::min(a.second, b.second) <= q.second &&
         q.second <= std::max(a.second, b.second);
}

bool on_boundary(const Point& p, int delta) {
  return p.first == 0 || p.second == 0 || p.first + p.second == delta;
}

Point parity(const Point& p) { return {p.first % 2, p.second % 2}; }

bool primitive(const Point& a, const Point& b) {
  return std::gcd(std::abs(a.first - b.first),
                  std::abs(a.second - b.second)) == 1;
}

bool same_side(const Point& a, const Point& b, int delta) {
  return (a.first == 0 && b.first == 0) || (a.second == 0 && b.second == 0) ||
         (a.first + a.second == delta && b.first + b.second == delta);
}

std::vector<Point> boundary_walk(int delta) {
  std::vector<Point> walk;
  walk.reserve(3 * delta);
  for (int k = 0; k < delta; ++k) walk.push_back({k, 0});
  for (int k = 0; k < delta; ++k) walk.push_back({delta - k, k});
  for (int k = 0; k < delta; ++k) walk.push_back({0, delta - k});
  return walk;
}

bool point_in_poly(const Point& q, const std::vector<Point>& poly) {
  const size_t n = poly.size();
  for (size_t k = 0; k < n; ++k) {
    const Point& a = poly[k];
    const Point& b = poly[(k + 1) % n];
    if (cross(a, b, q) == 0 && on_segment(a, b, q)) return true;
  }
  bool inside = false;
  for (size_t k = 0; k < n; ++k) {
    const Point& a = poly[k];
    const Point& b = poly[(k + 1) % n];
    if ((a.second > q.second) != (b.second > q.second)) {
      const int dy = b.second - a.second;
      const int num = (a.first - q.first) * dy +
                      (q.second - a.second) * (b.first - a.first);
      if (num * dy > 0) inside = !inside;
    }
  }
  return inside;
}

std::vector<Point> lattice_in_poly(const std::vector<Point>& poly, int delta) {
  std::vector<Point> out;
  for (const Point& p : lattice_points(delta))
    if (point_in_poly(p, poly)) out.push_back(p);
  return out;
}

std::vector<Point> zone_of(const Split& s, int delta) {
  const std::vector<Point> walk = boundary_walk(delta);
  const auto idx_of = [&](const Point& p) {
    return static_cast<size_t>(std::find(walk.begin(), walk.end(), p) -
                               walk.begin());
  };
  const size_t i0 = idx_of(s.front());
  const size_t i1 = idx_of(s.back());
  const auto arc = [&](size_t a, size_t b) {
    std::vector<Point> out;
    for (size_t k = a; k != b; k = (k + 1) % walk.size()) out.push_back(walk[k]);
    out.push_back(walk[b]);
    return out;
  };
  const std::vector<Point> arc1 = arc(i1, i0);
  const std::vector<Point> arc2 = arc(i0, i1);
  std::vector<Point> poly1(s.begin(), s.end() - 1);
  poly1.insert(poly1.end(), arc1.begin(), arc1.end() - 1);
  std::vector<Point> poly2(s.rbegin(), s.rend() - 1);
  poly2.insert(poly2.end(), arc2.begin(), arc2.end() - 1);
  const std::vector<Point> z1 = lattice_in_poly(poly1, delta);
  const std::vector<Point> z2 = lattice_in_poly(poly2, delta);
  const std::vector<Point> corners = {{0, 0}, {delta, 0}, {0, delta}};
  const auto corners_off_path = [&](const std::vector<Point>& z) {
    int c = 0;
    for (const Point& p : corners)
      if (std::find(z.begin(), z.end(), p) != z.end() &&
          std::find(s.begin(), s.end(), p) == s.end())
        ++c;
    return c;
  };
  return corners_off_path(z1) <= corners_off_path(z2) ? z1 : z2;
}

SignMask twist_mask(const Split& canon, const std::vector<Point>& zone,
                    int delta) {
  const Point a = canon.size() == 3 ? parity(canon[1]) : parity(canon[0]);
  const Point b = canon.size() == 3 ? parity(canon[0]) : parity(canon[1]);
  const int eps = (a.first * b.second + a.second * b.first) % 2;
  const int i = (a.second + b.second) % 2;
  const int j = (a.first + b.first) % 2;
  SignMask m = 0;
  for (const Point& q : zone)
    if ((eps + i * q.first + j * q.second) % 2)
      m |= SignMask{1} << flat_idx(delta, q);
  return m;
}

bool family_ok(Family f, const Point& pa, const Point& pb) {
  const bool has_zero = (pa == Point{0, 0}) || (pb == Point{0, 0});
  switch (f) {
    case Family::All: return true;
    case Family::WithZero: return has_zero;
    case Family::WithoutZero: return !has_zero;
  }
  return true;
}

void canonicalize(Split& s) {
  if (s.back() < s.front()) std::reverse(s.begin(), s.end());
}
}

std::vector<Split> raw_splits(int delta, Family family, bool drop_same_side,
                              bool drop_collinear) {
  if (delta < 1) throw std::invalid_argument("delta must be >= 1");
  if (delta > MAX_SUPPORTED_DELTA)
    throw std::invalid_argument(
        "delta > 9: a sign vector no longer fits in a uint64_t mask");
  const std::vector<Point> pts = lattice_points(delta);
  std::vector<Point> bpts;
  for (const Point& p : pts)
    if (on_boundary(p, delta)) bpts.push_back(p);

  std::vector<Split> out;

  for (size_t i = 0; i < bpts.size(); ++i)
    for (size_t j = i + 1; j < bpts.size(); ++j) {
      const Point &a = bpts[i], &b = bpts[j];
      if (parity(a) == parity(b)) continue;
      if (!primitive(a, b)) continue;
      if (drop_same_side && same_side(a, b, delta)) continue;
      if (!family_ok(family, parity(a), parity(b))) continue;
      out.push_back(Split{a, b});
    }

  for (const Point& q : pts) {
    if (on_boundary(q, delta)) continue;
    for (size_t i = 0; i < bpts.size(); ++i)
      for (size_t j = i + 1; j < bpts.size(); ++j) {
        const Point &a = bpts[i], &b = bpts[j];
        if (parity(a) != parity(b) || parity(a) == parity(q)) continue;
        if (!primitive(q, a) || !primitive(q, b)) continue;
        if (drop_collinear && cross(a, b, q) == 0) continue;
        if (!family_ok(family, parity(q), parity(a))) continue;
        out.push_back(Split{a, q, b});
      }
  }

  for (Split& s : out) canonicalize(s);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

SplitTable build_splits(int delta, Family family, bool drop_same_side,
                        bool drop_trivial, bool drop_collinear) {
  std::vector<Split> out =
      raw_splits(delta, family, drop_same_side, drop_collinear);

  std::vector<SignMask> twists;
  twists.reserve(out.size());
  for (const Split& s : out) twists.push_back(twist_mask(s, zone_of(s, delta),
                                                         delta));
  std::array<SignMask, 8> affine{};
  for (int k = 0; k < 8; ++k)
    affine[k] = point_mask(delta, [k](int x, int y) {
      return ((k >> 2) * x + ((k >> 1) & 1) * y + (k & 1)) % 2;
    });

  if (drop_trivial) {
    const SignMask interior = point_mask(delta, [delta](int x, int y) {
      return !on_boundary({x, y}, delta);
    });
    size_t keep = 0;
    for (size_t k = 0; k < out.size(); ++k) {
      bool trivial = false;
      for (SignMask m : affine)
        trivial = trivial || ((twists[k] ^ m) & interior) == 0;
      if (trivial) continue;
      if (keep != k) {
        out[keep] = std::move(out[k]);
        twists[keep] = twists[k];
      }
      ++keep;
    }
    out.resize(keep);
    twists.resize(keep);
  }

  SplitTable t;
  t.delta = delta;
  t.npoints = (delta + 1) * (delta + 2) / 2;
  t.harnack = point_mask(delta, [](int x, int y) {
    return ((x + 1) * (y + 1)) % 2;
  });
  t.affine = affine;
  t.splits = std::move(out);
  t.recs.reserve(t.splits.size());
  for (size_t k = 0; k < t.splits.size(); ++k) {
    const Split& s = t.splits[k];
    SplitRec r{};
    r.nsegs = static_cast<uint8_t>(s.size() - 1);
    r.seg_a = {std::min(s[0], s[1]), std::max(s[0], s[1])};
    r.idx_a = {flat_idx(delta, s[0]), flat_idx(delta, s[1])};
    if (r.nsegs == 2) {
      r.seg_b = {std::min(s[1], s[2]), std::max(s[1], s[2])};
      r.idx_b = {flat_idx(delta, s[1]), flat_idx(delta, s[2])};
    } else {
      r.seg_b = r.seg_a;
      r.idx_b = r.idx_a;
    }
    r.twist = twists[k];
    t.recs.push_back(r);
  }

  std::vector<std::vector<Split>> lower(delta);
  for (int k = 1; k < delta; ++k)
    lower[k] = raw_splits(delta - k, Family::All, drop_same_side, drop_collinear);
  for (size_t i = 0; i < t.splits.size(); ++i)
    for (int e = 0; e < 3; ++e) {
      int k = 0;
      while (k + 1 < delta) {
        Split moved = t.splits[i];
        for (Point& p : moved) {
          if (e == 0) p.first -= k + 1;
          if (e == 1) p.second -= k + 1;
        }
        canonicalize(moved);
        if (!std::binary_search(lower[k + 1].begin(), lower[k + 1].end(), moved))
          break;
        ++k;
      }
      t.recs[i].edge_dist[e] = static_cast<uint8_t>(k);
    }

  return t;
}

std::vector<Point> lattice_points(int delta) {
  std::vector<Point> pts;
  pts.reserve((delta + 1) * (delta + 2) / 2);
  for (int y = 0; y <= delta; ++y)
    for (int x = 0; x <= delta - y; ++x) pts.push_back({x, y});
  return pts;
}

std::vector<bool> unpack_signs(SignMask sig, int npoints) {
  std::vector<bool> out(npoints);
  for (int k = 0; k < npoints; ++k) out[k] = (sig >> k) & 1u;
  return out;
}

Profile profile_of(const SplitTable& table, const uint16_t* ids, int n) {
  if (n == 0) return kEmptyProfile;
  std::array<uint8_t, 3> d = table.recs[ids[0]].edge_dist;
  for (int i = 1; i < n; ++i)
    for (int e = 0; e < 3; ++e)
      d[e] = std::min(d[e], table.recs[ids[i]].edge_dist[e]);
  std::sort(d.begin(), d.end());
  return d[0] | (Profile{d[1]} << 8) | (Profile{d[2]} << 16);
}

int support_degree(int delta, Profile p) {
  if (p == kEmptyProfile) return 0;
  return delta - static_cast<int>((p & 0xff) + ((p >> 8) & 0xff) + (p >> 16));
}

std::string profile_text(Profile p) {
  if (p == kEmptyProfile) return "none";
  return std::to_string(p & 0xff) + ';' + std::to_string((p >> 8) & 0xff) +
         ';' + std::to_string(p >> 16);
}

const char* family_name(Family f) {
  switch (f) {
    case Family::All: return "all";
    case Family::WithZero: return "with-zero";
    case Family::WithoutZero: return "without-zero";
  }
  return "?";
}

bool parse_family(const std::string& s, Family& out) {
  if (s == "all") { out = Family::All; return true; }
  if (s == "with-zero") { out = Family::WithZero; return true; }
  if (s == "without-zero") { out = Family::WithoutZero; return true; }
  return false;
}
}
