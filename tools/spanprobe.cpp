#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "haas.h"
#include "isotopy_graph.h"

namespace {
using Isotopy::Point;
using Isotopy::Split;

int flat_idx(int delta, const Point& p) {
  return p.second * (delta + 1) - p.second * (p.second - 1) / 2 + p.first;
}

bool on_boundary(const Point& p, int delta) {
  return p.first == 0 || p.second == 0 || p.first + p.second == delta;
}

uint64_t pack(const std::vector<bool>& v) {
  uint64_t m = 0;
  for (size_t i = 0; i < v.size(); ++i)
    if (v[i]) m |= uint64_t{1} << i;
  return m;
}
}

const char* kUsage =
    "usage: spanprobe DELTA < FILE\n"
    "\n"
    "Decides whether the sign vector of a patchwork is reachable from\n"
    "Harnack's signs by Harnack splits whose segments are edges of its own\n"
    "triangulation, modulo the 8 affine masks.  By Haas' theorem this holds\n"
    "for every maximal T-curve.  Twists come from libisotopy's\n"
    "signs_from_splits only.  FILE has the format of pcomcheck.\n";

int main(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
    std::fputs(kUsage, stderr);
    return 2;
  }
  const int delta = std::atoi(argv[1]);
  const int npts = (delta + 1) * (delta + 2) / 2;

  std::string bits;
  std::cin >> bits;
  std::vector<Isotopy::Triangle> tris;
  int a, b, c;
  while (std::cin >> a >> b >> c) tris.push_back({a, b, c});

  uint64_t S = 0;
  for (size_t i = 0; i < bits.size(); ++i)
    if (bits[i] == '1') S |= uint64_t{1} << i;

  std::vector<Point> pt(npts);
  for (int y = 0; y <= delta; ++y)
    for (int x = 0; x + y <= delta; ++x) pt[flat_idx(delta, {x, y})] = {x, y};

  std::set<std::pair<int, int>> E;
  for (const auto& t : tris) {
    int v[3] = {t[0], t[1], t[2]};
    std::sort(v, v + 3);
    E.insert({v[0], v[1]});
    E.insert({v[0], v[2]});
    E.insert({v[1], v[2]});
  }
  std::printf("delta %d, %zu cells, %zu edges\n", delta, tris.size(), E.size());

  const uint64_t H = pack(Isotopy::signs_from_splits(delta, {}));
  std::printf("harnack 0x%llx  target 0x%llx\n",
              static_cast<unsigned long long>(H),
              static_cast<unsigned long long>(S));

  std::vector<std::pair<Split, uint64_t>> cands;
  auto try_split = [&](const Split& s) {
    try {
      const uint64_t tw = pack(Isotopy::signs_from_splits(delta, {s})) ^ H;
      cands.push_back({s, tw});
    } catch (const std::exception&) {
    }
  };
  for (const auto& [u, v] : E)
    if (on_boundary(pt[u], delta) && on_boundary(pt[v], delta))
      try_split({pt[u], pt[v]});
  std::map<int, std::vector<int>> adj;
  for (const auto& [u, v] : E) {
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
  for (const auto& [w, ns] : adj)
    for (size_t i = 0; i < ns.size(); ++i)
      for (size_t j = i + 1; j < ns.size(); ++j)
        if (on_boundary(pt[ns[i]], delta) && on_boundary(pt[ns[j]], delta))
          try_split({pt[ns[i]], pt[w], pt[ns[j]]});
  std::printf("valid splits inside this triangulation: %zu\n", cands.size());

  std::map<int, uint64_t> basis;
  for (const auto& [s, tw] : cands) {
    uint64_t cur = tw;
    while (cur) {
      const int p = 63 - __builtin_clzll(cur);
      auto it = basis.find(p);
      if (it == basis.end()) {
        basis[p] = cur;
        break;
      }
      cur ^= it->second;
    }
  }
  std::printf("twist span dimension: %zu (of %d)\n", basis.size(), npts);

  bool ok = false;
  for (int ai = 0; ai < 2; ++ai)
    for (int bi = 0; bi < 2; ++bi)
      for (int ci = 0; ci < 2; ++ci) {
        uint64_t m = 0;
        for (int i = 0; i < npts; ++i)
          if ((ai * pt[i].first + bi * pt[i].second + ci) % 2)
            m |= uint64_t{1} << i;
        uint64_t cur = S ^ H ^ m;
        while (cur) {
          const int p = 63 - __builtin_clzll(cur);
          auto it = basis.find(p);
          if (it == basis.end()) break;
          cur ^= it->second;
        }
        if (cur == 0) {
          ok = true;
          std::printf("  IN SPAN with affine mask (%d,%d,%d)\n", ai, bi, ci);
        }
      }
  if (!ok)
    std::printf(
        "  NOT in the span under any affine mask -- the patchwork's signs are\n"
        "  unreachable by splits inside its own triangulation\n");
  return 0;
}
