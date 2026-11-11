#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "compat.h"
#include "isotopy_graph.h"
#include "splits.h"

using SplitEnum::Point;
using SplitEnum::SignMask;
using SplitEnum::Split;
using SplitEnum::SplitTable;

namespace {
struct Side {
  std::vector<Point> poly;
  std::set<Point> pts;
  long area2 = 0;
};

struct Info {
  bool trivial = false;
  Side w;
  std::set<Point> b;
  std::set<Point> parities;
  Point apex{-1, -1};
  bool reflex = false;
};

bool on_boundary(const Point& p, int delta) {
  return p.first == 0 || p.second == 0 || p.first + p.second == delta;
}

bool corner(const Point& p, int delta) {
  return p == Point{0, 0} || p == Point{delta, 0} || p == Point{0, delta};
}

Point parity(const Point& p) { return {p.first % 2, p.second % 2}; }

int simplex_edge(const Point& p, int delta) {
  if (p.second == 0) return 0;
  if (p.first == 0) return 1;
  if (p.first + p.second == delta) return 2;
  return -1;
}

int edge_dist(const Point& p, int edge, int delta) {
  switch (edge) {
    case 0: return p.second;
    case 1: return p.first;
    default: return delta - p.first - p.second;
  }
}

std::vector<Point> boundary_walk(int delta) {
  std::vector<Point> walk;
  for (int k = 0; k < delta; ++k) walk.push_back({k, 0});
  for (int k = 0; k < delta; ++k) walk.push_back({delta - k, k});
  for (int k = 0; k < delta; ++k) walk.push_back({0, delta - k});
  return walk;
}

bool in_poly(const Point& q, const std::vector<Point>& poly, int scale) {
  const size_t n = poly.size();
  for (size_t k = 0; k < n; ++k) {
    const Point a{poly[k].first * scale, poly[k].second * scale};
    const Point b{poly[(k + 1) % n].first * scale, poly[(k + 1) % n].second * scale};
    if (SplitEnum::cross(a, b, q) == 0 &&
        std::min(a.first, b.first) <= q.first && q.first <= std::max(a.first, b.first) &&
        std::min(a.second, b.second) <= q.second &&
        q.second <= std::max(a.second, b.second))
      return true;
  }
  bool inside = false;
  for (size_t k = 0; k < n; ++k) {
    const Point a{poly[k].first * scale, poly[k].second * scale};
    const Point b{poly[(k + 1) % n].first * scale, poly[(k + 1) % n].second * scale};
    if ((a.second > q.second) != (b.second > q.second)) {
      const long dy = b.second - a.second;
      const long num = (a.first - q.first) * dy + (q.second - a.second) * (b.first - a.first);
      if (num * dy > 0) inside = !inside;
    }
  }
  return inside;
}

Side make_side(std::vector<Point> poly, int delta) {
  Side s;
  s.poly = std::move(poly);
  for (const Point& p : SplitEnum::lattice_points(delta))
    if (in_poly(p, s.poly, 1)) s.pts.insert(p);
  for (size_t k = 0; k < s.poly.size(); ++k) {
    const Point& a = s.poly[k];
    const Point& b = s.poly[(k + 1) % s.poly.size()];
    s.area2 += static_cast<long>(a.first) * b.second - static_cast<long>(a.second) * b.first;
  }
  s.area2 = std::labs(s.area2);
  return s;
}

std::array<Side, 2> sides_of(const Split& s, int delta) {
  const std::vector<Point> walk = boundary_walk(delta);
  const auto idx_of = [&](const Point& p) {
    return static_cast<size_t>(std::find(walk.begin(), walk.end(), p) - walk.begin());
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
  return {make_side(poly1, delta), make_side(poly2, delta)};
}

Info analyze(const Split& s, int delta, int& ambiguous) {
  Info info;
  for (const Point& p : s) info.parities.insert(parity(p));
  if (s.size() == 3) info.apex = s[1];
  const auto sides = sides_of(s, delta);
  int good = -1;
  for (int k = 0; k < 2; ++k) {
    bool ok = true;
    for (const Point& p : sides[k].pts)
      if (!on_boundary(p, delta) && !info.parities.count(parity(p))) ok = false;
    if (ok) {
      if (good >= 0) ++ambiguous;
      good = k;
    }
  }
  if (good < 0) return info;
  info.trivial = true;
  info.w = sides[good];
  for (const Point& p : info.w.pts)
    if (on_boundary(p, delta) && !info.parities.count(parity(p))) info.b.insert(p);
  if (s.size() == 3) {
    const Point a = s[0], p = s[1];
    const Point probe{4 * p.first + (p.first - a.first), 4 * p.second + (p.second - a.second)};
    info.reflex = in_poly(probe, info.w.poly, 4);
  }
  return info;
}

bool trivial_by_twist(const SplitTable& t, int k) {
  const SignMask interior = SplitEnum::point_mask(t.delta, [&](int x, int y) {
    return !on_boundary({x, y}, t.delta);
  });
  for (SignMask m : t.affine)
    if (((t.recs[k].twist ^ m) & interior) == 0) return true;
  return false;
}

struct Pairwise {
  long thin_bad = 0, shape_bad = 0, ambiguous = 0, reflex = 0;
  long c_boundary = 0, c_chord = 0, c_apex = 0, c_bad = 0;
  long trivial = 0;
};

Pairwise pairwise(const SplitTable& t, const std::vector<Info>& info) {
  Pairwise r;
  const int delta = t.delta;
  const int n = t.size();
  for (int i = 0; i < n; ++i) {
    if (info[i].trivial != trivial_by_twist(t, i)) ++r.shape_bad;
    if (!info[i].trivial) continue;
    ++r.trivial;
    if (info[i].reflex) ++r.reflex;
    for (const Point& p : info[i].w.pts)
      if (!on_boundary(p, delta) && p != info[i].apex) ++r.thin_bad;
    for (int j = 0; j < n; ++j) {
      if (j == i || !SplitEnum::splits_compatible(t.recs[i], t.recs[j])) continue;
      const Split& s = t.splits[j];
      for (int end = 0; end < 2; ++end) {
        const Point v = end ? s.back() : s.front();
        const Point w = end ? s[s.size() - 2] : s[1];
        if (!info[i].b.count(v)) continue;
        bool ok = info[i].w.pts.count(w) > 0;
        if (on_boundary(w, delta)) {
          const bool adjacent = simplex_edge(v, delta) >= 0 && s.size() == 2 &&
                                std::abs(v.first - w.first) + std::abs(v.second - w.second) == 1 &&
                                (v.first == w.first || v.second == w.second ||
                                 v.first + v.second == delta);
          if (adjacent) {
            ++r.c_boundary;
          } else {
            ++r.c_chord;
            ok = ok && s.size() == 2 && info[j].trivial &&
                 std::includes(info[i].w.pts.begin(), info[i].w.pts.end(),
                               info[j].w.pts.begin(), info[j].w.pts.end()) &&
                 info[j].w.area2 < info[i].w.area2 &&
                 (info[i].apex.first < 0 || !info[j].w.pts.count(info[i].apex));
          }
        } else {
          ++r.c_apex;
          const int e = simplex_edge(v, delta);
          ok = ok && w == info[i].apex && !corner(v, delta) && edge_dist(w, e, delta) == 1;
          for (const Point& c : info[i].b)
            if (corner(c, delta) && edge_dist(c, e, delta) == 0) ok = false;
        }
        if (!ok) {
          ++r.c_bad;
          std::printf("  violation: t=%d s=%d v=(%d,%d) w=(%d,%d)\n", i, j, v.first, v.second,
                      w.first, w.second);
        }
      }
    }
  }
  return r;
}

std::vector<bool> unpack(SignMask s, int n) {
  std::vector<bool> v(n);
  for (int i = 0; i < n; ++i) v[i] = (s >> i) & 1u;
  return v;
}

std::string scheme(const SplitTable& t, SignMask sig, const std::vector<Isotopy::Edge>& edges) {
  auto tris = Isotopy::refine(t.delta, edges);
  if (!tris) return "#REFINE";
  Isotopy::Graph g(t.delta, unpack(sig, t.npoints), *tris);
  return g.viro_notation();
}

struct NodeStats {
  long nodes = 0, peeled = 0, mismatches = 0;
};

void walk(const SplitTable& t, const std::vector<Info>& info,
          const std::vector<std::vector<char>>& compat, std::vector<int>& stack,
          std::vector<char>& cand, int maxdepth, NodeStats& st) {
  ++st.nodes;
  int best = -1;
  for (int k : stack)
    if (info[k].trivial && (best < 0 || info[k].w.area2 < info[best].w.area2)) best = k;
  if (best >= 0) {
    ++st.peeled;
    SignMask sig = t.harnack;
    std::vector<Isotopy::Edge> edges, rest;
    SignMask sig_rest = t.harnack;
    for (int k : stack) {
      sig ^= t.recs[k].twist;
      edges.push_back(t.recs[k].idx_a);
      if (t.recs[k].nsegs == 2) edges.push_back(t.recs[k].idx_b);
      if (k == best) continue;
      sig_rest ^= t.recs[k].twist;
      rest.push_back(t.recs[k].idx_a);
      if (t.recs[k].nsegs == 2) rest.push_back(t.recs[k].idx_b);
    }
    if (scheme(t, sig, edges) != scheme(t, sig_rest, rest)) {
      ++st.mismatches;
      std::string ids;
      for (int k : stack) ids += std::to_string(k) + " ";
      std::printf("  mismatch: node {%s} peeled %d\n", ids.c_str(), best);
    }
  }
  if (static_cast<int>(stack.size()) >= maxdepth) return;
  const int n = t.size();
  const int from = stack.empty() ? 0 : stack.back() + 1;
  for (int j = from; j < n; ++j) {
    if (!cand[j]) continue;
    std::vector<char> next(cand);
    for (int k = 0; k < n; ++k) next[k] = next[k] && compat[j][k];
    stack.push_back(j);
    walk(t, info, compat, stack, next, maxdepth, st);
    stack.pop_back();
  }
}
}

const char* kUsage =
    "usage: peelaudit DELTA [MAXDEPTH]\n"
    "\n"
    "Audits the trivial-split filter on the split table of degree DELTA,\n"
    "default 5.  Pairwise, for every trivial split t and every compatible s\n"
    "with an endpoint in the side of t: that side has no lattice point off\n"
    "the boundary except the apex of t, and the edge of s at that endpoint\n"
    "stays in the side and is a boundary edge, a chord making s trivial with\n"
    "a strictly smaller side, or the edge from the endpoint to the apex of t\n"
    "at lattice distance 1.  Node by node up to MAXDEPTH, default unlimited:\n"
    "removing the trivial member with the smallest side keeps the scheme.\n";

int main(int argc, char** argv) {
  if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    std::fputs(kUsage, stdout);
    return 0;
  }
  const int delta = argc > 1 ? std::atoi(argv[1]) : 5;
  const int maxdepth = argc > 2 ? std::atoi(argv[2]) : 0;
  for (int same_side_keep = 0; same_side_keep < 2; ++same_side_keep) {
    const SplitTable t = SplitEnum::build_splits(delta, SplitEnum::Family::All, !same_side_keep,
                                                 false, false);
    std::vector<Info> info;
    int ambiguous = 0;
    for (const Split& s : t.splits) info.push_back(analyze(s, delta, ambiguous));
    const Pairwise r = pairwise(t, info);
    std::printf(
        "delta %d same-side %s: %d splits, %ld trivial, %ld reflex sides, %d ambiguous sides\n"
        "  shape mismatches %ld, thin violations %ld\n"
        "  edges at B_t: boundary %ld, chord %ld, apex %ld, violations %ld\n",
        delta, same_side_keep ? "keep" : "drop", t.size(), r.trivial, r.reflex, ambiguous,
        r.shape_bad, r.thin_bad, r.c_boundary, r.c_chord, r.c_apex, r.c_bad);
    if (maxdepth > 0 && !same_side_keep) {
      const int n = t.size();
      std::vector<std::vector<char>> compat(n, std::vector<char>(n));
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          compat[i][j] = i != j && SplitEnum::splits_compatible(t.recs[i], t.recs[j]);
      NodeStats total;
#pragma omp parallel
      {
        NodeStats st;
#pragma omp for schedule(dynamic)
        for (int j = 0; j < n; ++j) {
          std::vector<int> stack{j};
          std::vector<char> cand(compat[j].begin(), compat[j].end());
          walk(t, info, compat, stack, cand, maxdepth, st);
        }
#pragma omp critical
        {
          total.nodes += st.nodes;
          total.peeled += st.peeled;
          total.mismatches += st.mismatches;
        }
      }
      std::printf("  nodes to depth %d: %ld, with a trivial member %ld, scheme mismatches %ld\n",
                  maxdepth, total.nodes + 1, total.peeled, total.mismatches);
      std::fflush(stdout);
    }
  }
  return 0;
}
