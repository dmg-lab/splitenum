#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <chrono>
#include <filesystem>
#include <thread>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "chunkio.h"
#include "compat.h"
#include "enumerate.h"
#include "gpuclassify.h"
#include "jobs.h"
#include "master.h"
#include "net.h"
#include "tcp.h"
#include "pool.h"
#include "splits.h"
#include "store.h"
#include "symmetry.h"
#include "walk.h"

using SplitEnum::Compat;
using SplitEnum::Family;
using SplitEnum::SignMask;
using SplitEnum::SplitTable;
using SplitEnum::Symmetry;
using SplitEnum::build_compat;
using SplitEnum::build_splits;
using SplitEnum::unpack_signs;

constexpr int kTopDelta = MAX_DELTA < 9 ? MAX_DELTA : 9;

namespace {
const Isotopy::Split B{{0, 1}, {5, 3}};
const Isotopy::Split D1{{3, 5}, {1, 2}, {5, 3}};
const Isotopy::Split D2a{{0, 3}, {3, 5}};
const Isotopy::Split D2b{{0, 7}, {3, 5}};

bool table_contains(const SplitTable& t, Isotopy::Split s) {
  if (s.back() < s.front()) std::reverse(s.begin(), s.end());
  return std::find(t.splits.begin(), t.splits.end(), s) != t.splits.end();
}
}

TEST_CASE("split table sizes at delta 8", "[splits]") {
  REQUIRE(build_splits(8, Family::All, true, false, true).size() == 1227);
  REQUIRE(build_splits(8, Family::WithZero, true, false, true).size() == 1038);
  REQUIRE(build_splits(8, Family::WithoutZero, true, false, true).size() == 189);
  REQUIRE(build_splits(8, Family::All, false, false, true).size() == 1251);
}

TEST_CASE("split table is canonical, sorted and duplicate-free", "[splits]") {
  const SplitTable t = build_splits(8, Family::All, true, false, true);
  for (const Isotopy::Split& s : t.splits) REQUIRE(s.front() < s.back());
  REQUIRE(std::is_sorted(t.splits.begin(), t.splits.end()));
  REQUIRE(std::set<Isotopy::Split>(t.splits.begin(), t.splits.end()).size() ==
          t.splits.size());
  REQUIRE(table_contains(t, B));
  REQUIRE(table_contains(t, D1));
  REQUIRE(table_contains(t, D2a));
  REQUIRE(table_contains(t, D2b));
  REQUIRE_FALSE(table_contains(t, Isotopy::Split{{0, 3}, {3, 5}, {0, 7}}));
}

TEST_CASE("every enumerated split is valid to libisotopy", "[splits]") {
  for (int d : {3, 4, 5, 8}) {
    const SplitTable t = build_splits(d, Family::All, true, false, true);
    for (const Isotopy::Split& s : t.splits)
      REQUIRE_NOTHROW(Isotopy::compatible(d, {s}));
  }
}

TEST_CASE("twist masks reproduce signs_from_splits", "[splits]") {
  for (int d : {3, 4, 5, 6, 7, 8}) {
    const SplitTable t = build_splits(d, Family::All, true, false, true);
    REQUIRE(unpack_signs(t.harnack, t.npoints) ==
            Isotopy::signs_from_splits(d, {}));
    for (int i = 0; i < t.size(); ++i)
      REQUIRE(unpack_signs(t.harnack ^ t.recs[i].twist, t.npoints) ==
              Isotopy::signs_from_splits(d, {t.splits[i]}));
  }
}

TEST_CASE("incremental XOR equals signs_from_splits on a collection",
          "[splits]") {
  const SplitTable t = build_splits(8, Family::All, true, false, true);
  const Compat c = build_compat(t);
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> pick(0, t.size() - 1);
  int checked = 0;
  while (checked < 200) {
    std::vector<int> coll;
    for (int tries = 0; tries < 60 && coll.size() < 5; ++tries) {
      const int v = pick(rng);
      bool ok = std::find(coll.begin(), coll.end(), v) == coll.end();
      for (int u : coll) ok = ok && c.get(u, v);
      if (ok) coll.push_back(v);
    }
    if (coll.size() < 2) continue;
    SplitEnum::SignMask sig = t.harnack;
    std::vector<Isotopy::Split> ss;
    for (int v : coll) {
      sig ^= t.recs[v].twist;
      ss.push_back(t.splits[v]);
    }
    REQUIRE(unpack_signs(sig, t.npoints) == Isotopy::signs_from_splits(8, ss));
    ++checked;
  }
}

TEST_CASE("compatibility really is pairwise", "[compat]") {
  const SplitTable t = build_splits(8, Family::All, true, false, true);
  const Compat c = build_compat(t);
  std::mt19937 rng(777);
  std::uniform_int_distribution<int> pick(0, t.size() - 1);
  for (int k = 0; k < 100000; ++k) {
    const int size = 2 + (k % 3);
    std::vector<int> ids;
    while (static_cast<int>(ids.size()) < size) {
      const int v = pick(rng);
      if (std::find(ids.begin(), ids.end(), v) == ids.end()) ids.push_back(v);
    }
    bool pairwise = true;
    for (size_t a = 0; a < ids.size() && pairwise; ++a)
      for (size_t b = a + 1; b < ids.size() && pairwise; ++b)
        pairwise = c.get(ids[a], ids[b]);
    std::vector<Isotopy::Split> ss;
    for (int v : ids) ss.push_back(t.splits[v]);
    REQUIRE(Isotopy::compatible(8, ss) == pairwise);
  }
}

TEST_CASE("compat matrix matches the documented examples", "[compat]") {
  const SplitTable t = build_splits(8, Family::All, true, false, true);
  const Compat c = build_compat(t);
  const auto id = [&](Isotopy::Split s) {
    if (s.back() < s.front()) std::reverse(s.begin(), s.end());
    const auto it = std::find(t.splits.begin(), t.splits.end(), s);
    REQUIRE(it != t.splits.end());
    return static_cast<int>(it - t.splits.begin());
  };
  REQUIRE(c.get(id(B), id(D1)));
  REQUIRE(c.get(id(B), id(D2a)));
  REQUIRE(c.get(id(B), id(D2b)));
  REQUIRE(c.get(id(D1), id(D2a)));
  REQUIRE(c.get(id(D1), id(D2b)));
  REQUIRE(c.get(id(D2a), id(D2b)));
  const Isotopy::Split crossing{{2, 0}, {0, 5}};
  REQUIRE_FALSE(c.get(id(B), id(crossing)));
  const Isotopy::Split shares_edge{{5, 3}, {1, 2}, {7, 1}};
  REQUIRE_FALSE(c.get(id(D1), id(shares_edge)));
}

TEST_CASE("known scheme through the same path the search uses", "[emit]") {
  const SplitTable t = build_splits(8, Family::All, true, false, true);
  const auto rec_of = [&](Isotopy::Split s) {
    if (s.back() < s.front()) std::reverse(s.begin(), s.end());
    const auto it = std::find(t.splits.begin(), t.splits.end(), s);
    REQUIRE(it != t.splits.end());
    return t.recs[it - t.splits.begin()];
  };
  const SplitEnum::SplitRec rb = rec_of(B);
  std::vector<Isotopy::Edge> edges{rb.idx_a};
  auto tris = Isotopy::refine(8, edges);
  REQUIRE(tris.has_value());
  Isotopy::Graph g(8, unpack_signs(t.harnack ^ rb.twist, t.npoints), *tris);
  REQUIRE(g.viro_notation() == "<10v1<11>>");

  const SplitEnum::SplitRec r1 = rec_of(D1), r2a = rec_of(D2a), r2b = rec_of(D2b);
  edges = {rb.idx_a, r1.idx_a, r1.idx_b, r2a.idx_a, r2b.idx_a};
  tris = Isotopy::refine(8, edges);
  REQUIRE(tris.has_value());
  Isotopy::Graph gs(8,
                    unpack_signs(t.harnack ^ rb.twist ^ r1.twist ^ r2a.twist ^
                                     r2b.twist,
                                 t.npoints),
                    *tris);
  REQUIRE(gs.viro_notation() == "<17v1<2v1<1>>>");
}

TEST_CASE("empty collection is the Harnack curve", "[emit]") {
  const SplitTable t = build_splits(8, Family::All, true, false, true);
  auto tris = Isotopy::refine(8, {});
  REQUIRE(tris.has_value());
  Isotopy::Graph g(8, unpack_signs(t.harnack, t.npoints), *tris);
  REQUIRE(g.viro_notation() == "<18v1<3>>");
  REQUIRE(g.is_mcurve());
}

TEST_CASE("delta above 9 is rejected rather than silently truncated",
          "[splits]") {
  REQUIRE_THROWS_AS(build_splits(10, Family::All, true, false, true), std::invalid_argument);
}

namespace {
namespace fs = std::filesystem;

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE(f.good());
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

SplitEnum::Options test_options(int delta, const std::string& out) {
  SplitEnum::Options o;
  o.delta = delta;
  o.out = out;
  o.cli = "test";
  o.symmetry = false;
  o.drop_trivial = false;
  o.drop_collinear = true;
  return o;
}

std::string sorted_types_csv(const std::string& path) {
  std::ifstream f(path);
  REQUIRE(f.good());
  std::string header, line;
  std::getline(f, header);
  std::vector<std::string> rows;
  while (std::getline(f, line))
    if (!line.empty()) rows.push_back(line);
  std::sort(rows.begin(), rows.end());
  std::string out = header + '\n';
  for (const std::string& r : rows) out += r + '\n';
  return out;
}

std::map<std::string, uint64_t> viro_totals(const std::string& path) {
  std::ifstream f(path);
  REQUIRE(f.good());
  std::string line;
  std::getline(f, line);
  std::map<std::string, uint64_t> out;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const size_t q = line.find("\",");
    size_t pos = q + 2;
    for (int field = 0; field < 3; ++field) pos = line.find(',', pos) + 1;
    pos = line.find("\",", pos) + 2;
    pos = line.find(',', pos) + 1;
    out[line.substr(1, q - 1)] += std::stoull(line.substr(pos));
  }
  return out;
}

const std::string kTestOut = "bin/testout";
}

TEST_CASE("chunked run reproduces the documented degree-4 tree", "[run]") {
  fs::remove_all(kTestOut);
  const auto s = SplitEnum::run(test_options(4, kTestOut + "/ref"));
  REQUIRE(s.total_nodes == 112344);
  REQUIRE(s.max_depth_reached == 10);
  REQUIRE(s.compat_pairs == 1314);
  REQUIRE(s.classified == 112344);
  REQUIRE(s.walk_complete);
  REQUIRE_FALSE(s.stopped);
  REQUIRE_FALSE(fs::exists(kTestOut + "/ref/sigs.csv"));
  REQUIRE_FALSE(fs::exists(kTestOut + "/ref/classify"));
  REQUIRE(viro_totals(kTestOut + "/ref/types.csv") ==
          std::map<std::string, uint64_t>{{"<4>", 112344}});
}

TEST_CASE("output is invariant under chunk size and sharding", "[run]") {
  const std::string ref_types = sorted_types_csv(kTestOut + "/ref/types.csv");

  auto tiny = test_options(4, kTestOut + "/tiny");
  tiny.chunk_pairs = 7;
  SplitEnum::run(tiny);
  REQUIRE(sorted_types_csv(kTestOut + "/tiny/types.csv") == ref_types);

  auto s0 = test_options(4, kTestOut + "/shard0");
  s0.shard_index = 0;
  s0.shard_count = 2;
  const auto r0 = SplitEnum::run(s0);
  REQUIRE_FALSE(r0.walk_complete);
  auto s1 = test_options(4, kTestOut + "/shard1");
  s1.shard_index = 1;
  s1.shard_count = 2;
  SplitEnum::run(s1);
  for (const auto& e : fs::directory_iterator(kTestOut + "/shard1/chunks"))
    fs::copy_file(e.path(),
                  kTestOut + "/shard0/chunks/" + e.path().filename().string());
  auto sc = test_options(4, kTestOut + "/shard0");
  sc.phase = SplitEnum::Phase::Classify;
  const auto rc = SplitEnum::run(sc);
  REQUIRE(rc.total_nodes == 112344);
  REQUIRE(sorted_types_csv(kTestOut + "/shard0/types.csv") == ref_types);
}

TEST_CASE("both phases resume from a killed run's files", "[run]") {
  const std::string ref_types = sorted_types_csv(kTestOut + "/ref/types.csv");
  const std::string dir = kTestOut + "/restart";

  auto opt = test_options(4, dir);
  opt.chunk_pairs = 16;
  SplitEnum::run(opt);
  const std::string before = slurp(dir + "/types.csv");
  REQUIRE(sorted_types_csv(dir + "/types.csv") == ref_types);

  int k = 0;
  for (const auto& e : fs::directory_iterator(dir + "/chunks"))
    if (++k % 3 == 0) fs::remove(e.path());
  std::ofstream(dir + "/chunks/walk-c000000.bin.tmp") << "garbage";
  fs::remove(dir + "/types.csv");

  opt.phase = SplitEnum::Phase::Walk;
  const auto w = SplitEnum::run(opt);
  REQUIRE(w.walk_complete);
  REQUIRE_FALSE(fs::exists(dir + "/types.csv"));

  opt.phase = SplitEnum::Phase::Classify;
  const auto s = SplitEnum::run(opt);
  REQUIRE(s.total_nodes == 112344);
  REQUIRE(slurp(dir + "/types.csv") == before);
}

TEST_CASE("a directory refuses chunks from another configuration", "[run]") {
  const std::string dir = kTestOut + "/ref";
  REQUIRE_THROWS_WITH(SplitEnum::run(test_options(3, dir)),
                      Catch::Contains("different configuration"));
}

namespace {
std::vector<uint64_t> chunk_depth_totals(const std::string& dir) {
  std::vector<uint64_t> out;
  for (const auto& e : fs::directory_iterator(dir + "/chunks")) {
    const auto h = SplitEnum::read_chunk_header(e.path().string());
    for (size_t d = 0; d < h.nodes_per_depth.size(); ++d) {
      if (out.size() <= d) out.resize(d + 1, 0);
      out[d] += h.nodes_per_depth[d];
    }
  }
  return out;
}
}

TEST_CASE("symmetry group structure across degrees and families",
          "[symmetry]") {
  for (int d : {3, 4, 5, 6, 7, 8})
    REQUIRE(Symmetry(build_splits(d, Family::All, true, false, true)).order() == 6);
  REQUIRE(Symmetry(build_splits(8, Family::All, false, false, true)).order() == 6);
  for (Family f : {Family::WithZero, Family::WithoutZero}) {
    REQUIRE(Symmetry(build_splits(8, f, true, false, true)).order() == 6);
    REQUIRE(Symmetry(build_splits(6, f, true, false, true)).order() == 6);
    REQUIRE(Symmetry(build_splits(5, f, true, false, true)).order() == 2);
    REQUIRE(Symmetry(build_splits(3, f, true, false, true)).order() == 2);
  }
}

TEST_CASE("symmetry permutations preserve compatibility", "[symmetry]") {
  const SplitTable t = build_splits(5, Family::All, true, false, true);
  const Compat c = build_compat(t);
  const Symmetry sym(t);
  REQUIRE(sym.elems().size() == 5);
  for (const auto& e : sym.elems()) {
    bool same = true;
    for (int i = 0; i < t.size() && same; ++i)
      for (int j = i + 1; j < t.size() && same; ++j)
        same = c.get(i, j) == c.get(e.split_perm[i], e.split_perm[j]);
    REQUIRE(same);
  }
}

TEST_CASE("symmetry images of a collection share a scheme", "[symmetry]") {
  const SplitTable t = build_splits(5, Family::All, true, false, true);
  const Compat c = build_compat(t);
  const Symmetry sym(t);
  std::mt19937 rng(4242);
  std::uniform_int_distribution<int> pick(0, t.size() - 1);
  std::uniform_int_distribution<int> pickg(
      0, static_cast<int>(sym.elems().size()) - 1);
  int checked = 0;
  while (checked < 500) {
    std::vector<int> coll;
    for (int tries = 0; tries < 60 && coll.size() < 5; ++tries) {
      const int v = pick(rng);
      bool ok = std::find(coll.begin(), coll.end(), v) == coll.end();
      for (int u : coll) ok = ok && c.get(u, v);
      if (ok) coll.push_back(v);
    }
    if (coll.empty()) continue;
    const auto& e = sym.elems()[pickg(rng)];
    std::vector<uint16_t> ids, img_ids;
    SignMask sig = t.harnack, img = t.harnack;
    for (int v : coll) {
      ids.push_back(static_cast<uint16_t>(v));
      img_ids.push_back(e.split_perm[v]);
      sig ^= t.recs[v].twist;
      img ^= t.recs[e.split_perm[v]].twist;
    }
    std::sort(ids.begin(), ids.end());
    std::sort(img_ids.begin(), img_ids.end());
    const auto a = SplitEnum::classify_sig(t, sig, ids.data(), ids.size());
    const auto b =
        SplitEnum::classify_sig(t, img, img_ids.data(), img_ids.size());
    REQUIRE_FALSE(a.err);
    REQUIRE(a.viro == b.viro);
    REQUIRE(a.p == b.p);
    REQUIRE(a.n == b.n);
    REQUIRE(a.mcurve == b.mcurve);
    ++checked;
  }
}

TEST_CASE("symmetry-on run reproduces the exact degree-4 totals and types",
          "[symmetry]") {
  auto on = test_options(4, kTestOut + "/on4");
  on.symmetry = true;
  const auto s = SplitEnum::run(on);
  REQUIRE(s.total_nodes == 112344);
  REQUIRE(s.max_depth_reached == 10);
  REQUIRE(s.sym_order == 6);
  REQUIRE(chunk_depth_totals(kTestOut + "/on4") ==
          chunk_depth_totals(kTestOut + "/ref"));
  REQUIRE(sorted_types_csv(kTestOut + "/on4/types.csv") ==
          sorted_types_csv(kTestOut + "/ref/types.csv"));
  REQUIRE(s.classified < 112344);
  REQUIRE(s.classified * 6 >= 112344);
}

TEST_CASE("symmetry-on output is invariant under chunking and resume",
          "[symmetry]") {
  const std::string ref_types = sorted_types_csv(kTestOut + "/on4/types.csv");
  const std::string dir = kTestOut + "/on4tiny";

  auto tiny = test_options(4, dir);
  tiny.symmetry = true;
  tiny.chunk_pairs = 7;
  SplitEnum::run(tiny);
  REQUIRE(sorted_types_csv(dir + "/types.csv") == ref_types);
  const std::string before = slurp(dir + "/types.csv");

  int k = 0;
  for (const auto& e : fs::directory_iterator(dir + "/chunks"))
    if (++k % 3 == 0) fs::remove(e.path());
  fs::remove(dir + "/types.csv");
  SplitEnum::run(tiny);
  REQUIRE(slurp(dir + "/types.csv") == before);
}

TEST_CASE("symmetry-on and -off directories refuse to mix", "[symmetry]") {
  REQUIRE_THROWS_WITH(SplitEnum::run(test_options(4, kTestOut + "/on4")),
                      Catch::Contains("different configuration"));
}

TEST_CASE("degree-5 count-only totals match with symmetry on", "[.][d5]") {
  auto opt = test_options(5, kTestOut + "/d5on");
  opt.symmetry = true;
  opt.count_only = true;
  REQUIRE(SplitEnum::run(opt).total_nodes == 32715916ull);
}

TEST_CASE("inverse permutations and allowed rows", "[symmetry]") {
  for (int d : {4, 5}) {
    const SplitTable t = build_splits(d, Family::All, true, false, true);
    const Symmetry sym(t);
    const int n = t.size();
    REQUIRE(sym.words() == (n + 63) / 64);
    for (const auto& e : sym.elems())
      for (int i = 0; i < n; ++i) REQUIRE(e.inv_perm[e.split_perm[i]] == i);
    for (int u = 0; u < n; ++u) {
      const uint64_t* row = sym.allowed_row(u);
      for (int w = 0; w < n; ++w) {
        bool ok = w >= u;
        for (const auto& e : sym.elems()) ok = ok && e.split_perm[w] >= u;
        REQUIRE(((row[w >> 6] >> (w & 63)) & 1u) == (ok ? 1u : 0u));
      }
    }
  }
}

TEST_CASE("masked canonical agrees with the full test", "[symmetry]") {
  for (int d : {4, 5}) {
    const SplitTable t = build_splits(d, Family::All, true, false, true);
    const Compat c = build_compat(t);
    const Symmetry sym(t);
    const int n = t.size();
    std::mt19937 rng(31337);
    std::uniform_int_distribution<int> pick(0, n - 1);
    std::vector<uint16_t> scratch(16);
    int inside = 0, outside = 0, active_hits = 0;
    while (inside < 2000 || outside < 200 || active_hits < 50) {
      std::vector<uint16_t> s;
      for (int tries = 0; tries < 80 && s.size() < 6; ++tries) {
        const int v = pick(rng);
        bool ok = std::find(s.begin(), s.end(), v) == s.end();
        for (uint16_t u : s) ok = ok && c.get(u, v);
        if (ok) s.push_back(static_cast<uint16_t>(v));
      }
      if (s.size() < 2) continue;
      std::sort(s.begin(), s.end());
      const int u0 = s[0];
      const uint64_t* row = sym.allowed_row(u0);
      bool in_row = true;
      for (uint16_t v : s) in_row = in_row && ((row[v >> 6] >> (v & 63)) & 1u);
      uint32_t wf = 0;
      const bool full = sym.canonical_stack(s.data(),
                                            static_cast<int>(s.size()),
                                            scratch.data(), wf);
      if (!in_row) {
        REQUIRE(!full);
        ++outside;
        continue;
      }
      unsigned active = 0;
      for (size_t i = 0; i < sym.elems().size(); ++i) {
        const uint16_t tgt = sym.elems()[i].inv_perm[u0];
        if (std::find(s.begin(), s.end(), tgt) != s.end())
          active |= 1u << i;
      }
      if (active == 0) {
        REQUIRE(full);
        REQUIRE(wf == static_cast<uint32_t>(sym.order()));
      } else {
        uint32_t wm = 0;
        const bool masked = sym.canonical_stack_active(
            s.data(), static_cast<int>(s.size()), scratch.data(), active, wm);
        REQUIRE(masked == full);
        if (full) REQUIRE(wm == wf);
        ++active_hits;
      }
      ++inside;
    }
  }
}

TEST_CASE("trivial splits are dropped by the mask-modulo criterion",
          "[splits]") {
  REQUIRE(build_splits(8, Family::All, true, true, true).size() == 849);
  REQUIRE(build_splits(7, Family::All, true, true, true).size() == 426);
  REQUIRE(build_splits(6, Family::All, true, true, true).size() == 210);
  REQUIRE(build_splits(5, Family::All, true, true, true).size() == 72);
  REQUIRE(build_splits(4, Family::All, true, true, true).size() == 0);

  const SplitTable full = build_splits(5, Family::All, true, false, true);
  const SplitTable f = build_splits(5, Family::All, true, true, true);
  REQUIRE(table_contains(full, Isotopy::Split{{0, 1}, {1, 0}}));
  REQUIRE_FALSE(table_contains(f, Isotopy::Split{{0, 1}, {1, 0}}));
  REQUIRE_FALSE(table_contains(f, Isotopy::Split{{1, 0}, {0, 3}}));
  REQUIRE(table_contains(f, Isotopy::Split{{0, 1}, {3, 2}}));
  for (int i = 0; i < f.size(); ++i) {
    const auto it =
        std::find(full.splits.begin(), full.splits.end(), f.splits[i]);
    REQUIRE(it != full.splits.end());
    REQUIRE(f.recs[i].twist ==
            full.recs[it - full.splits.begin()].twist);
  }
}

TEST_CASE("filtered table keeps the whole S_3", "[splits][symmetry]") {
  for (int d : {5, 6, 7, 8})
    REQUIRE(Symmetry(build_splits(d, Family::All, true, true, true)).order() == 6);
}

TEST_CASE("empty filtered table is rejected", "[splits]") {
  auto opt = test_options(4, kTestOut + "/triv-empty");
  opt.drop_trivial = true;
  REQUIRE_THROWS_WITH(SplitEnum::run(opt), Catch::Contains("no splits"));
}

TEST_CASE("published degree-7 realization classifies as <Jv3v1<11>>",
          "[regression]") {
  const std::string bits = "111000101000000101100001000000010000";
  const std::vector<Isotopy::Triangle> tris = {
      {0, 1, 8}, {1, 2, 10}, {1, 8, 9}, {1, 9, 17}, {1, 10, 17},
      {2, 3, 11}, {2, 10, 18}, {2, 11, 25}, {2, 18, 25}, {3, 4, 12},
      {3, 11, 25}, {3, 12, 19}, {3, 19, 25}, {4, 5, 12}, {5, 6, 13},
      {5, 12, 19}, {5, 13, 25}, {5, 19, 25}, {6, 7, 14}, {6, 13, 25},
      {6, 14, 20}, {6, 20, 25}, {8, 9, 17}, {8, 15, 16}, {8, 16, 17},
      {10, 17, 18}, {15, 16, 23}, {15, 21, 22}, {15, 22, 29}, {15, 23, 29},
      {16, 17, 23}, {17, 18, 25}, {17, 23, 29}, {17, 24, 25}, {17, 24, 29},
      {21, 22, 29}, {21, 26, 27}, {21, 27, 28}, {21, 28, 29}, {24, 25, 29},
      {26, 27, 30}, {27, 28, 30}, {28, 29, 30}, {29, 30, 31}, {29, 31, 33},
      {29, 32, 33}, {30, 31, 33}, {32, 33, 34}, {33, 34, 35}
  };
  REQUIRE(bits.size() == 36);
  REQUIRE(tris.size() == 49);
  std::vector<bool> signs;
  for (const char c : bits) signs.push_back(c == '1');

  Isotopy::Graph g(7, signs, tris);
  REQUIRE(g.viro_notation() == "<Jv3v1<11>>");
  REQUIRE(g.p_regions == 12);
  REQUIRE(g.n_regions == 4);
  REQUIRE(g.is_mcurve());
}

TEST_CASE("collinear doubles: table growth and membership", "[splits][collinear]") {
  REQUIRE(build_splits(5, Family::All, true, false, false).size() == 171);
  REQUIRE(build_splits(6, Family::All, true, false, false).size() == 390);
  REQUIRE(build_splits(7, Family::All, true, false, false).size() == 690);
  REQUIRE(build_splits(8, Family::All, true, false, false).size() == 1254);
  REQUIRE(build_splits(5, Family::All, true, true, false).size() == 78);
  REQUIRE(build_splits(6, Family::All, true, true, false).size() == 222);
  REQUIRE(build_splits(7, Family::All, true, true, false).size() == 444);
  REQUIRE(build_splits(8, Family::All, true, true, false).size() == 873);
  REQUIRE(build_splits(4, Family::All, true, true, false).size() == 0);

  const SplitTable t7 = build_splits(7, Family::All, true, true, false);
  const Isotopy::Split cd1{{0, 1}, {2, 2}, {4, 3}};
  const Isotopy::Split cd2{{1, 0}, {2, 2}, {3, 4}};
  REQUIRE(table_contains(t7, cd1));
  REQUIRE(table_contains(t7, cd2));
  const SplitTable t7_old = build_splits(7, Family::All, true, true, true);
  REQUIRE_FALSE(table_contains(t7_old, cd1));
  REQUIRE_FALSE(table_contains(build_splits(7, Family::All, true, false, false),
                               Isotopy::Split{{0, 1}, {0, 2}, {0, 3}}));
}

TEST_CASE("collinear doubles: twist masks reproduce signs_from_splits",
          "[splits][collinear]") {
  for (int d : {5, 6, 7}) {
    const SplitTable t = build_splits(d, Family::All, true, false, false);
    for (int i = 0; i < t.size(); ++i)
      REQUIRE(unpack_signs(t.harnack ^ t.recs[i].twist, t.npoints) ==
              Isotopy::signs_from_splits(d, {t.splits[i]}));
  }
}

TEST_CASE("collinear doubles: symmetry group survives", "[symmetry][collinear]") {
  for (int d : {5, 6, 7, 8})
    REQUIRE(Symmetry(build_splits(d, Family::All, true, true, false)).order() ==
            6);
}

TEST_CASE("collinear doubles: the jv3v1_11 collection classifies <Jv3v1<11>>",
          "[emit][collinear]") {
  const SplitTable t = build_splits(7, Family::All, true, true, false);
  const std::vector<Isotopy::Split> coll = {
      {{0, 1}, {2, 2}, {4, 3}}, {{1, 0}, {2, 2}, {3, 4}},
      {{2, 0}, {4, 3}},         {{3, 0}, {4, 3}},
      {{4, 3}, {5, 0}},         {{0, 2}, {3, 4}},
      {{0, 3}, {3, 4}}};
  const Compat c = build_compat(t);
  std::vector<int> ids;
  SplitEnum::SignMask sig = t.harnack;
  std::vector<Isotopy::Edge> edges;
  for (const Isotopy::Split& s : coll) {
    REQUIRE(table_contains(t, s));
    const int id = static_cast<int>(
        std::find(t.splits.begin(), t.splits.end(), s) - t.splits.begin());
    for (const int u : ids) REQUIRE(c.get(u, id));
    ids.push_back(id);
    sig ^= t.recs[id].twist;
    edges.push_back(t.recs[id].idx_a);
    if (t.recs[id].nsegs == 2) edges.push_back(t.recs[id].idx_b);
  }
  auto tris = Isotopy::refine(7, edges);
  REQUIRE(tris.has_value());
  Isotopy::Graph g(7, unpack_signs(sig, t.npoints), *tris);
  REQUIRE(g.viro_notation() == "<Jv3v1<11>>");
  REQUIRE(g.p_regions == 12);
  REQUIRE(g.n_regions == 4);
  REQUIRE(g.is_mcurve());
}

TEST_CASE("collinear doubles: delta-4 walk stays a single type", "[run][collinear]") {
  SplitEnum::Options o = test_options(4, kTestOut + "/ck4");
  o.drop_collinear = false;
  const SplitEnum::RunStats s = SplitEnum::run(o);
  REQUIRE(s.nsplits == 78);
  REQUIRE(s.total_nodes == 217236);
  REQUIRE(viro_totals(kTestOut + "/ck4/types.csv") ==
          std::map<std::string, uint64_t>{{"<4>", 217236}});
}

TEST_CASE("every split is a Harnack split of the paper", "[splits][harnack]") {
  const auto on_boundary = [](const Isotopy::Point& p, int d) {
    return p.first == 0 || p.second == 0 || p.first + p.second == d;
  };
  for (const Family family : {Family::All, Family::WithZero, Family::WithoutZero})
    for (int d = 3; d <= kTopDelta; ++d)
      for (bool same_side : {true, false})
        for (bool collinear : {true, false}) {
          INFO("delta " << d << " family " << SplitEnum::family_name(family)
                        << " same_side " << same_side << " collinear " << collinear);
          const std::vector<Isotopy::Split> raw =
              SplitEnum::raw_splits(d, family, same_side, collinear);
          const SplitTable t = build_splits(d, family, same_side, false, collinear);
          REQUIRE(t.splits == raw);
          for (const Isotopy::Split& s : raw) {
            REQUIRE(on_boundary(s.front(), d));
            REQUIRE(on_boundary(s.back(), d));
            if (s.size() == 3) REQUIRE_FALSE(on_boundary(s[1], d));
          }
        }

  REQUIRE(build_splits(5, Family::All, true, true, false).size() == 78);
  REQUIRE(build_splits(6, Family::All, true, true, false).size() == 222);
  REQUIRE(build_splits(7, Family::All, true, true, false).size() == 444);
  REQUIRE(build_splits(8, Family::All, true, true, false).size() == 873);
  for (int d : {5, 6, 7, 8})
    REQUIRE(Symmetry(build_splits(d, Family::All, true, true, false)).order() == 6);

  const SplitTable t7 = build_splits(7, Family::All, true, true, false);
  REQUIRE(table_contains(t7, Isotopy::Split{{0, 1}, {2, 2}, {4, 3}}));
  REQUIRE(table_contains(t7, Isotopy::Split{{1, 0}, {2, 2}, {3, 4}}));
}

namespace {
using SplitEnum::Job;
using SplitEnum::JobFold;
using SplitEnum::JobResult;

constexpr uint64_t kNoBudget = UINT64_MAX;

template <typename F>
JobFold drain(const SplitEnum::Walk& w, uint64_t budget, std::vector<Job> queue,
              F&& on_result) {
  SplitEnum::Ctx c;
  c.init(w);
  JobFold fold;
  while (!queue.empty()) {
    Job j = std::move(queue.back());
    queue.pop_back();
    JobResult r = SplitEnum::execute_job(c, w, j, budget);
    on_result(j, r);
    fold.fold(r);
    for (Job& x : r.rest) queue.push_back(std::move(x));
  }
  return fold;
}

JobFold drain(const SplitEnum::Walk& w, uint64_t budget,
              std::vector<Job> queue) {
  return drain(w, budget, std::move(queue), [](const Job&, const JobResult&) {});
}

std::vector<uint64_t> trimmed(std::vector<uint64_t> v) {
  while (!v.empty() && v.back() == 0) v.pop_back();
  return v;
}

SplitEnum::Options d5_default(const std::string& out) {
  SplitEnum::Options o;
  o.delta = 5;
  o.out = out;
  o.cli = "test";
  return o;
}

std::string fold_types_csv(const JobFold& f, const std::string& path,
                           int delta) {
  f.types.write_csv(path, delta);
  return sorted_types_csv(path);
}
}

TEST_CASE("budgeted jobs partition the tree exactly, count-only", "[jobs]") {
  for (bool sym : {false, true}) {
    const std::string dir = kTestOut + "/jobs-count-" + std::to_string(sym);
    fs::remove_all(dir);
    auto opt = test_options(4, dir);
    opt.symmetry = sym;
    opt.count_only = true;
    const auto ref = SplitEnum::run(opt);
    REQUIRE(ref.total_nodes == 112344);
    const SplitEnum::Walk w = SplitEnum::prepare(opt);
    for (uint64_t budget : {uint64_t{1}, uint64_t{2}, uint64_t{7},
                            uint64_t{1000}, kNoBudget}) {
      INFO("symmetry " << sym << " budget " << budget);
      const JobFold f = drain(w, budget, {SplitEnum::root_job(w)});
      REQUIRE(f.total_nodes() == 112344);
      REQUIRE(trimmed(f.nodes_per_depth) == trimmed(ref.nodes_per_depth));
      if (budget == kNoBudget) REQUIRE(f.jobs == 1);
    }
  }
}

TEST_CASE("budgeted jobs reproduce types.csv", "[jobs]") {
  {
    const std::string dir = kTestOut + "/jobs-types4";
    fs::remove_all(dir);
    const auto opt = test_options(4, dir);
    SplitEnum::run(opt);
    const SplitEnum::Walk w = SplitEnum::prepare(opt);
    const JobFold f = drain(w, 997, {SplitEnum::root_job(w)});
    REQUIRE(f.total_nodes() == 112344);
    REQUIRE(fold_types_csv(f, dir + "/jobs-types.csv", 4) ==
            sorted_types_csv(dir + "/types.csv"));
  }
  const std::string dir = kTestOut + "/jobs-types5";
  fs::remove_all(dir);
  const auto opt = d5_default(dir);
  const auto ref = SplitEnum::run(opt);
  const SplitEnum::Walk w = SplitEnum::prepare(opt);
  REQUIRE(w.sym);
  for (uint64_t budget : {uint64_t{97}, uint64_t{997}}) {
    INFO("budget " << budget);
    const JobFold f = drain(w, budget, {SplitEnum::root_job(w)});
    REQUIRE(f.jobs > 1);
    REQUIRE(f.total_nodes() == ref.total_nodes);
    REQUIRE(trimmed(f.nodes_per_depth) == trimmed(ref.nodes_per_depth));
    REQUIRE(fold_types_csv(f, dir + "/jobs-types.csv", 5) ==
            sorted_types_csv(dir + "/types.csv"));
  }
}

TEST_CASE("remainder jobs are nonempty, in range and canonical", "[jobs]") {
  auto opt = d5_default(kTestOut + "/jobs-rest");
  opt.count_only = true;
  const SplitEnum::Walk w = SplitEnum::prepare(opt);
  const int n = w.table.size();
  std::vector<uint16_t> scratch(n + 2);
  std::mt19937 rng(20260916);
  uint64_t checked = 0;
  for (int round = 0; round < 4; ++round) {
    const uint64_t budget = 1 + rng() % 40;
    INFO("budget " << budget);
    drain(w, budget, {SplitEnum::root_job(w)},
          [&](const Job& j, const JobResult& r) {
      std::set<size_t> levels;
      for (const Job& x : r.rest) {
        REQUIRE(x.prefix.size() >= j.prefix.size());
        levels.insert(x.prefix.size());
      }
      REQUIRE(levels.size() == r.rest.size());
      for (const Job& x : r.rest) {
        REQUIRE_FALSE(x.root);
        REQUIRE(x.lo < x.hi);
        REQUIRE(x.hi <= n);
        REQUIRE(std::adjacent_find(x.prefix.begin(), x.prefix.end(),
                                   std::greater_equal<>()) == x.prefix.end());
        if (!x.prefix.empty()) REQUIRE(x.lo > x.prefix.back());
        uint32_t weight = 0;
        REQUIRE(w.sym->canonical_stack(x.prefix.data(),
                                       static_cast<int>(x.prefix.size()),
                                       scratch.data(), weight));
        const auto cand = SplitEnum::job_candidates(w, x);
        REQUIRE_FALSE(cand.empty());
        REQUIRE(cand.front() == x.lo);
        ++checked;
      }
    });
  }
  REQUIRE(checked > 100);
}

TEST_CASE("split_job partitions a job", "[jobs]") {
  auto opt = d5_default(kTestOut + "/jobs-split");
  opt.count_only = true;
  const SplitEnum::Walk w = SplitEnum::prepare(opt);
  const JobFold whole = drain(w, kNoBudget, {SplitEnum::root_job(w)});
  for (int parts : {1, 3, 17}) {
    INFO("parts " << parts);
    const auto pieces = SplitEnum::split_job(w, SplitEnum::root_job(w), parts);
    REQUIRE(static_cast<int>(pieces.size()) == parts);
    REQUIRE(std::count_if(pieces.begin(), pieces.end(),
                          [](const Job& j) { return j.root; }) == 1);
    const JobFold f = drain(w, 500, pieces);
    REQUIRE(f.nodes_per_depth == whole.nodes_per_depth);
  }

  std::vector<Job> deep;
  drain(w, 50, {SplitEnum::root_job(w)},
        [&](const Job&, const JobResult& r) {
    for (const Job& x : r.rest)
      if (x.prefix.size() >= 2 &&
          SplitEnum::job_candidates(w, x).size() >= 2)
        deep.push_back(x);
  });
  REQUIRE_FALSE(deep.empty());
  for (size_t i = 0; i < deep.size(); i += 1 + deep.size() / 16) {
    const JobFold one = drain(w, kNoBudget, {deep[i]});
    const auto pieces = SplitEnum::split_job(w, deep[i], 4);
    REQUIRE(pieces.size() ==
            std::min<size_t>(4, SplitEnum::job_candidates(w, deep[i]).size()));
    const JobFold split = drain(w, kNoBudget, pieces);
    REQUIRE(split.nodes_per_depth == one.nodes_per_depth);
  }
}

TEST_CASE("budgeted jobs respect max-depth", "[jobs]") {
  for (int cap : {1, 2, 3}) {
    INFO("max depth " << cap);
    const std::string dir = kTestOut + "/jobs-cap" + std::to_string(cap);
    fs::remove_all(dir);
    auto opt = test_options(4, dir);
    opt.count_only = true;
    opt.max_depth = cap;
    const auto ref = SplitEnum::run(opt);
    const SplitEnum::Walk w = SplitEnum::prepare(opt);
    const JobFold f = drain(w, 5, {SplitEnum::root_job(w)});
    REQUIRE(f.total_nodes() == ref.total_nodes);
    REQUIRE(trimmed(f.nodes_per_depth) == trimmed(ref.nodes_per_depth));
  }
}

TEST_CASE("an aborted job yields its whole remainder", "[jobs]") {
  auto opt = d5_default(kTestOut + "/jobs-abort");
  opt.count_only = true;
  const SplitEnum::Walk w = SplitEnum::prepare(opt);
  const JobFold whole = drain(w, kNoBudget, {SplitEnum::root_job(w)});

  const std::atomic<bool> stop{true};
  SplitEnum::Ctx c;
  c.init(w);
  c.abort = &stop;
  const JobResult r = SplitEnum::execute_job(c, w, SplitEnum::root_job(w), 1000);
  REQUIRE(r.nodes() == 0);
  REQUIRE(r.rest == std::vector<Job>{SplitEnum::root_job(w)});

  c.abort = nullptr;
  const JobResult first =
      SplitEnum::execute_job(c, w, SplitEnum::root_job(w), 1000);
  REQUIRE_FALSE(first.finished());
  c.abort = &stop;
  std::vector<Job> resumed;
  for (const Job& j : first.rest) {
    const JobResult a = SplitEnum::execute_job(c, w, j, 1000);
    REQUIRE(a.nodes() == 0);
    REQUIRE_FALSE(a.rest.empty());
    for (const Job& x : a.rest) resumed.push_back(x);
  }
  JobFold total = drain(w, 1000, resumed);
  total.fold(first);
  REQUIRE(total.nodes_per_depth == whole.nodes_per_depth);
}

TEST_CASE("job messages round-trip and refuse other configurations",
          "[jobs]") {
  const auto off_opt = test_options(4, kTestOut + "/jobs-wire-off");
  const auto on_opt = d5_default(kTestOut + "/jobs-wire-on");
  for (const SplitEnum::Options* opt : {&off_opt, &on_opt}) {
    const SplitEnum::Walk w = SplitEnum::prepare(*opt);
    const uint64_t h = SplitEnum::job_config_hash(w);
    SplitEnum::Ctx c;
    c.init(w);
    const JobResult r = SplitEnum::execute_job(c, w, SplitEnum::root_job(w), 1000);
    REQUIRE_FALSE(r.rest.empty());
    REQUIRE_FALSE(r.types.empty());

    const std::vector<uint8_t> jb = SplitEnum::encode_job(r.rest.front(), h);
    REQUIRE(SplitEnum::decode_job(jb, h) == r.rest.front());
    REQUIRE_THROWS(SplitEnum::decode_job(jb, h + 1));
    REQUIRE_THROWS(SplitEnum::decode_job(
        std::vector<uint8_t>(jb.begin(), jb.end() - 1), h));

    const std::vector<uint8_t> rb = SplitEnum::encode_result(r, h);
    const JobResult d = SplitEnum::decode_result(rb, h);
    REQUIRE(d.nodes_per_depth == r.nodes_per_depth);
    REQUIRE(d.classified == r.classified);
    REQUIRE(d.errs == r.errs);
    REQUIRE(d.classify_calls == r.classify_calls);
    REQUIRE(d.rest == r.rest);
    REQUIRE(d.types.size() == r.types.size());
    for (size_t i = 0; i < d.types.size(); ++i) {
      REQUIRE(d.types[i].viro == r.types[i].viro);
      REQUIRE(d.types[i].p == r.types[i].p);
      REQUIRE(d.types[i].n == r.types[i].n);
      REQUIRE(d.types[i].mcurve == r.types[i].mcurve);
      REQUIRE(d.types[i].count == r.types[i].count);
      REQUIRE(d.types[i].first_depth == r.types[i].first_depth);
      REQUIRE(d.types[i].witness == r.types[i].witness);
    }
    REQUIRE_THROWS(SplitEnum::decode_result(rb, h ^ 1));
    REQUIRE_THROWS(SplitEnum::decode_result(
        std::vector<uint8_t>(rb.begin(), rb.end() - 3), h));
    REQUIRE_THROWS(SplitEnum::decode_job(rb, h));
  }
  const SplitEnum::Walk a = SplitEnum::prepare(off_opt);
  auto sym_opt = off_opt;
  sym_opt.symmetry = true;
  const SplitEnum::Walk b = SplitEnum::prepare(sym_opt);
  REQUIRE(SplitEnum::job_config_hash(a) != SplitEnum::job_config_hash(b));
}

TEST_CASE("degree-6 budgeted jobs reproduce types.csv", "[.][d6][jobs]") {
  const std::string dir = kTestOut + "/jobs-types6";
  fs::remove_all(dir);
  auto opt = d5_default(dir);
  opt.delta = 6;
  const auto ref = SplitEnum::run(opt);
  const SplitEnum::Walk w = SplitEnum::prepare(opt);
  const JobFold f = drain(w, 1000, {SplitEnum::root_job(w)});
  REQUIRE(f.jobs > 1);
  REQUIRE(f.total_nodes() == ref.total_nodes);
  REQUIRE(trimmed(f.nodes_per_depth) == trimmed(ref.nodes_per_depth));
  REQUIRE(fold_types_csv(f, dir + "/jobs-types.csv", 6) ==
          sorted_types_csv(dir + "/types.csv"));
}

namespace {
std::array<uint8_t, 3> edge_dist_of(const SplitTable& t, Isotopy::Split s) {
  if (s.back() < s.front()) std::reverse(s.begin(), s.end());
  const auto it = std::find(t.splits.begin(), t.splits.end(), s);
  REQUIRE(it != t.splits.end());
  return t.recs[it - t.splits.begin()].edge_dist;
}

void brute_force_support(const SplitEnum::Options& opt, const std::string& dir) {
  const SplitTable t = build_splits(opt.delta, opt.family, opt.drop_same_side,
                                    opt.drop_trivial, opt.drop_collinear);
  const Compat c = build_compat(t);
  SplitEnum::TypeTable ref;
  std::vector<uint16_t> stack;
  uint64_t nodes = 0;
  const auto node = [&](SignMask sig) {
    const auto res =
        SplitEnum::classify_sig(t, sig, stack.data(), stack.size());
    ref.add(res.viro, res.p, res.n, res.mcurve,
            SplitEnum::profile_of(t, stack.data(),
                                  static_cast<int>(stack.size())),
            1, static_cast<int>(stack.size()), stack.data(), stack.size());
    ++nodes;
  };
  const auto dfs = [&](auto&& self, int from, SignMask sig) -> void {
    node(sig);
    for (int v = from; v < t.size(); ++v) {
      if (!std::all_of(stack.begin(), stack.end(),
                       [&](uint16_t u) { return c.get(u, v); }))
        continue;
      stack.push_back(static_cast<uint16_t>(v));
      self(self, v + 1, sig ^ t.recs[v].twist);
      stack.pop_back();
    }
  };
  dfs(dfs, 0, t.harnack);
  fs::create_directories(dir);
  ref.write_csv(dir + "/brute.csv", opt.delta);
  const std::string want = sorted_types_csv(dir + "/brute.csv");
  for (int pairs : {1, 7}) {
    INFO("chunk_pairs " << pairs);
    auto o = opt;
    o.chunk_pairs = pairs;
    o.out = dir + "/run-" + std::to_string(pairs);
    fs::remove_all(o.out);
    const auto st = SplitEnum::run(o);
    REQUIRE(st.total_nodes == nodes);
    REQUIRE(sorted_types_csv(o.out + "/types.csv") == want);
  }
}
}

TEST_CASE("edge distances match lower-degree split tables", "[splits][support]") {
  for (const Family family : {Family::All, Family::WithZero})
  for (int delta = 3; delta <= kTopDelta; ++delta)
    for (bool same_side : {true, false}) {
      INFO("delta " << delta << " drop_same_side " << same_side);
      const SplitTable t =
          build_splits(delta, family, same_side, false, false);
      std::vector<std::vector<Isotopy::Split>> lower(delta);
      for (int k = 1; k < delta; ++k)
        lower[k] = SplitEnum::raw_splits(delta - k, Family::All, same_side, false);
      for (int i = 0; i < t.size(); ++i)
        for (int e = 0; e < 3; ++e)
          for (int k = 1; k < delta; ++k) {
            Isotopy::Split moved = t.splits[i];
            for (auto& p : moved) {
              if (e == 0) p.first -= k;
              if (e == 1) p.second -= k;
            }
            if (moved.back() < moved.front())
              std::reverse(moved.begin(), moved.end());
            const bool fits = std::binary_search(lower[k].begin(),
                                                 lower[k].end(), moved);
            REQUIRE(fits == (k <= t.recs[i].edge_dist[e]));
          }
    }
  const SplitTable t8 = build_splits(8, Family::All, true, true, false);
  REQUIRE(edge_dist_of(t8, {{0, 0}, {1, 3}, {2, 0}}) ==
          std::array<uint8_t, 3>{0, 0, 3});
  const SplitTable t7 = build_splits(7, Family::All, true, true, false);
  REQUIRE(edge_dist_of(t7, {{0, 0}, {1, 3}, {6, 0}}) ==
          std::array<uint8_t, 3>{0, 0, 1});
}

TEST_CASE("symmetries permute the edge distances", "[symmetry][support]") {
  for (const Family family : {Family::All, Family::WithZero, Family::WithoutZero})
  for (int delta = 5; delta <= kTopDelta; ++delta) {
    INFO("delta " << delta << " family " << SplitEnum::family_name(family));
    const SplitTable t = build_splits(delta, family, true, true, false);
    const Symmetry sym(t);
    REQUIRE(sym.elems().size() == (family == Family::All || delta % 2 == 0 ? 5u : 1u));
    for (const auto& g : sym.elems()) {
      std::array<int, 3> pi{0, 1, 2};
      bool found = false;
      do {
        bool ok = true;
        for (int s = 0; s < t.size() && ok; ++s)
          for (int e = 0; e < 3 && ok; ++e)
            ok = t.recs[g.split_perm[s]].edge_dist[pi[e]] == t.recs[s].edge_dist[e];
        found = found || ok;
      } while (!found && std::next_permutation(pi.begin(), pi.end()));
      REQUIRE(found);
    }
  }
}

TEST_CASE("support profiles match a brute-force walk at delta 5",
          "[run][support]") {
  auto opt = d5_default(kTestOut + "/support5");
  brute_force_support(opt, kTestOut + "/support5");
}

TEST_CASE("support profiles match a brute-force walk under a family filter",
          "[run][support]") {
  auto opt = d5_default(kTestOut + "/support6-family");
  opt.delta = 6;
  opt.family = Family::WithZero;
  brute_force_support(opt, kTestOut + "/support6-family");
}

TEST_CASE("support profiles match a brute-force walk at delta 6",
          "[.][d6][support]") {
  auto opt = d5_default(kTestOut + "/support6");
  opt.delta = 6;
  brute_force_support(opt, kTestOut + "/support6");
}

TEST_CASE("exact key classifies every visited node", "[run][support]") {
  auto opt = d5_default(kTestOut + "/exact-calls");
  fs::remove_all(opt.out);
  const auto s = SplitEnum::run(opt);
  auto off = opt;
  off.symmetry = false;
  off.out = kTestOut + "/exact-calls-off";
  fs::remove_all(off.out);
  const auto t = SplitEnum::run(off);
  REQUIRE(t.classified == t.total_nodes);
  REQUIRE(s.classified < s.total_nodes);
  REQUIRE(sorted_types_csv(opt.out + "/types.csv") ==
          sorted_types_csv(off.out + "/types.csv"));
}

namespace {
std::vector<uint16_t> random_collection(const SplitTable& t, const Compat& c,
                                        std::mt19937& rng, size_t target) {
  std::vector<uint16_t> ids;
  for (int tries = 0; tries < 4 * t.size() && ids.size() < target; ++tries) {
    const uint16_t v = static_cast<uint16_t>(rng() % t.size());
    if (std::all_of(ids.begin(), ids.end(),
                    [&](uint16_t u) { return u != v && c.get(u, v); }))
      ids.push_back(v);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}
}

TEST_CASE("lean refine reproduces Isotopy::refine", "[refine]") {
  for (int delta = 4; delta <= kTopDelta; ++delta) {
    INFO("delta " << delta);
    const SplitTable t = build_splits(delta, Family::All, true, false, false);
    const Compat c = build_compat(t);
    const SplitEnum::LeanRefine lean(t);
    REQUIRE(lean.nedges() == 3 * delta * (delta + 1) / 2);
    REQUIRE(lean.matches_library(t, nullptr, 0));
    std::mt19937 rng(delta);
    for (int i = 0; i < 2000; ++i) {
      const auto ids = random_collection(t, c, rng, 1 + rng() % 40);
      uint64_t acc[SplitEnum::LeanRefine::kWords];
      REQUIRE(lean.refine(ids.data(), static_cast<int>(ids.size()), acc));
      REQUIRE(lean.matches_library(t, ids.data(), static_cast<int>(ids.size())));
    }
  }
}

TEST_CASE("lean refine rejects crossing forced edges like the library",
          "[refine]") {
  const SplitTable t = build_splits(6, Family::All, true, false, false);
  const Compat c = build_compat(t);
  const SplitEnum::LeanRefine lean(t);
  int rejected = 0;
  for (int i = 0; i < t.size(); ++i)
    for (int j = i + 1; j < t.size(); j += 7) {
      if (c.get(i, j)) continue;
      const uint16_t ids[2] = {static_cast<uint16_t>(i), static_cast<uint16_t>(j)};
      uint64_t acc[SplitEnum::LeanRefine::kWords];
      rejected += !lean.refine(ids, 2, acc);
      REQUIRE(lean.matches_library(t, ids, 2));
    }
  REQUIRE(rejected > 0);
}

TEST_CASE("the warp sweep reproduces LeanRefine byte for byte", "[refine]") {
  for (int delta = 4; delta <= kTopDelta; ++delta) {
    INFO("delta " << delta);
    const SplitTable t = build_splits(delta, Family::All, true, false, false);
    const Compat c = build_compat(t);
    const SplitEnum::LeanRefine lean(t);
    const SplitEnum::RefineTables rt = SplitEnum::refine_tables(lean);
    const size_t nbytes = 2 * static_cast<size_t>(lean.nedges());
    std::vector<uint8_t> want(nbytes), got(nbytes);
    std::mt19937 rng(100 + delta);
    int refined = 0, failed = 0;
    for (int i = 0; i < 10000; ++i) {
      std::vector<uint16_t> ids;
      if (i % 2 == 0) {
        ids = random_collection(t, c, rng, 1 + rng() % 40);
      } else {
        ids.resize(1 + rng() % 8);
        for (uint16_t& id : ids) id = static_cast<uint16_t>(rng() % t.size());
      }
      const int n = static_cast<int>(ids.size());
      uint64_t acc[SplitEnum::LeanRefine::kWords];
      const bool cpu = lean.refine(ids.data(), n, acc);
      SplitEnum::RefineScratch rs;
      const bool warp = SplitEnum::sweep_warp(rt, ids.data(), n, rs,
                                              IsotopyGPU::Warp<1>{});
      REQUIRE(cpu == warp);
      if (!cpu) {
        ++failed;
        continue;
      }
      ++refined;
      lean.edges(acc, want.data());
      SplitEnum::expand_edges(rt, rs, got.data(), IsotopyGPU::Warp<1>{});
      REQUIRE(got == want);
    }
    REQUIRE(refined > 4000);
    REQUIRE(failed > 0);
  }
}

TEST_CASE("the walk stack carries the lean refine seed and classifies like "
          "classify_sig", "[refine]") {
  for (int delta : {5, 6}) {
    INFO("delta " << delta);
    auto opt = d5_default(kTestOut + "/lean-seed");
    opt.delta = delta;
    const SplitEnum::Walk w = SplitEnum::prepare(opt);
    REQUIRE(w.lean);
    SplitEnum::Ctx ctx;
    ctx.init(w);
    ctx.reset();
    constexpr int kW = SplitEnum::LeanRefine::kWords;
    std::mt19937 rng(delta);
    for (int i = 0; i < 500; ++i) {
      const auto ids = random_collection(w.table, w.compat, rng, rng() % 20);
      for (uint16_t id : ids) ctx.push(id);
      std::vector<uint64_t> want(2 * kW, 0);
      for (uint16_t id : ids)
        for (int k = 0; k < kW; ++k) {
          want[k] |= w.lean->forced_row(id)[k];
          want[kW + k] |= w.lean->blocked_row(id)[k];
        }
      const uint64_t* seed = ctx.lean_seed();
      REQUIRE(std::equal(want.begin(), want.end(), seed));

      const auto ref =
          SplitEnum::classify_sig(w.table, ctx.sig, ids.data(), ids.size());
      const auto got =
          SplitEnum::classify_lean(w.table, *w.lean, ctx.sig, seed, seed + kW);
      REQUIRE_FALSE(ref.err);
      REQUIRE(got.viro == ref.viro);
      REQUIRE(got.p == ref.p);
      REQUIRE(got.n == ref.n);
      REQUIRE(got.mcurve == ref.mcurve);
      REQUIRE(got.err == ref.err);
      for (auto it = ids.rbegin(); it != ids.rend(); ++it) ctx.pop(*it);
    }
    REQUIRE(ctx.sig == w.table.harnack);

    std::vector<uint16_t> seq;
    for (int i = 0; i < 5000; ++i) {
      const size_t cut = seq.empty() ? 0 : rng() % (seq.size() + 1);
      while (seq.size() > cut) {
        ctx.pop(seq.back());
        seq.pop_back();
      }
      for (size_t grow = rng() % 4; grow > 0 && seq.size() < 20; --grow) {
        seq.push_back(static_cast<uint16_t>(rng() % w.table.size()));
        ctx.push(seq.back());
      }
      if (rng() % 3 == 0) continue;
      std::vector<uint64_t> want(2 * kW, 0);
      for (uint16_t id : seq)
        for (int k = 0; k < kW; ++k) {
          want[k] |= w.lean->forced_row(id)[k];
          want[kW + k] |= w.lean->blocked_row(id)[k];
        }
      REQUIRE(std::equal(want.begin(), want.end(), ctx.lean_seed()));
    }
  }
}

namespace {
class FakeRanks : public SplitEnum::SliceRunner {
 public:
  FakeRanks(int ranks, int threads, uint64_t budget, int stop_after)
      : ranks_(ranks), threads_(threads), budget_(budget), stop_after_(stop_after) {}

  void start(const SplitEnum::Walk& w) override {
    walk_ = &w;
    hash_ = SplitEnum::job_config_hash(w);
    for (int r = 0; r < ranks_; ++r)
      pools_.push_back(std::make_unique<SplitEnum::Pool>(w, threads_, budget_));
  }
  int slots() const override { return ranks_; }

  void submit(int slot, std::vector<Job> jobs, double seconds) override {
    SplitEnum::JobBatch batch;
    batch.jobs = std::move(jobs);
    batch.seconds = seconds;
    inbox_.emplace_back(slot, SplitEnum::encode_jobs(batch, hash_));
  }

  int wait(JobResult& out) override {
    REQUIRE_FALSE(inbox_.empty());
    const auto [slot, bytes] = inbox_.front();
    inbox_.erase(inbox_.begin());
    SplitEnum::JobBatch batch = SplitEnum::decode_jobs(bytes, hash_);
    const JobResult r =
        pools_[slot]->run(std::move(batch.jobs), batch.seconds, nullptr);
    out = SplitEnum::decode_result(SplitEnum::encode_result(r, hash_), hash_);
    if (++waits_ == stop_after_) SplitEnum::stop_flag().store(true);
    return slot;
  }

  int waits() const { return waits_; }

 private:
  int ranks_, threads_;
  uint64_t budget_;
  int stop_after_;
  int waits_ = 0;
  const SplitEnum::Walk* walk_ = nullptr;
  uint64_t hash_ = 0;
  std::vector<std::unique_ptr<SplitEnum::Pool>> pools_;
  std::vector<std::pair<int, std::vector<uint8_t>>> inbox_;
};

SplitEnum::Options sched_options(SplitEnum::Options o, const std::string& out) {
  o.out = out;
  o.scheduler_jobs = true;
  o.slice_seconds = 0.002;
  o.checkpoint_seconds = 0;
  fs::remove_all(out);
  return o;
}
}

TEST_CASE("job scheduler reproduces the chunked count-only walk", "[sched]") {
  auto ref_opt = test_options(4, kTestOut + "/sched-count-ref");
  ref_opt.count_only = true;
  fs::remove_all(ref_opt.out);
  const auto ref = SplitEnum::run(ref_opt);
  REQUIRE(ref.total_nodes == 112344);
  for (int threads : {1, 4})
    for (uint64_t budget : {uint64_t{50}, uint64_t{1000}, uint64_t{100000}}) {
      INFO("threads " << threads << " budget " << budget);
      auto opt = sched_options(ref_opt, kTestOut + "/sched-count");
      opt.threads = threads;
      opt.budget = budget;
      const auto s = SplitEnum::run(opt);
      REQUIRE_FALSE(s.stopped);
      REQUIRE(s.scheduler_jobs);
      REQUIRE(s.jobs_done > 0);
      REQUIRE(s.total_nodes == ref.total_nodes);
      REQUIRE(s.nodes_per_depth == ref.nodes_per_depth);
    }
}

TEST_CASE("job scheduler reproduces the chunked types.csv", "[sched]") {
  auto ref_opt = d5_default(kTestOut + "/sched-types-ref");
  fs::remove_all(ref_opt.out);
  const auto ref = SplitEnum::run(ref_opt);
  const std::string want = sorted_types_csv(ref_opt.out + "/types.csv");
  for (int threads : {1, 4})
    for (uint64_t budget : {uint64_t{1}, uint64_t{1000}, uint64_t{100000}}) {
      INFO("threads " << threads << " budget " << budget);
      auto opt = sched_options(ref_opt, kTestOut + "/sched-types");
      opt.threads = threads;
      opt.budget = budget;
      const auto s = SplitEnum::run(opt);
      REQUIRE_FALSE(s.stopped);
      REQUIRE(s.total_nodes == ref.total_nodes);
      REQUIRE(s.nodes_per_depth == ref.nodes_per_depth);
      REQUIRE(sorted_types_csv(opt.out + "/types.csv") == want);
      const auto again = SplitEnum::run(opt);
      REQUIRE(again.total_nodes == ref.total_nodes);
      REQUIRE(sorted_types_csv(opt.out + "/types.csv") == want);
    }
}

TEST_CASE("job scheduler resumes from its checkpoint across fake ranks",
          "[sched]") {
  auto ref_opt = d5_default(kTestOut + "/sched-resume-ref");
  fs::remove_all(ref_opt.out);
  const auto ref = SplitEnum::run(ref_opt);
  const std::string want = sorted_types_csv(ref_opt.out + "/types.csv");
  for (int ranks : {1, 3}) {
    INFO("ranks " << ranks);
    auto opt = sched_options(ref_opt, kTestOut + "/sched-resume");
    opt.budget = 50;
    int rounds = 0;
    SplitEnum::RunStats s;
    do {
      FakeRanks runner(ranks, 1, opt.budget, 2);
      s = SplitEnum::run(opt, &runner);
      ++rounds;
      REQUIRE(fs::exists(opt.out + "/jobs/state.bin"));
      if (s.stopped) REQUIRE_FALSE(fs::exists(opt.out + "/types.csv"));
      REQUIRE(rounds < 100000);
    } while (s.stopped);
    REQUIRE(rounds > 1);
    REQUIRE(s.total_nodes == ref.total_nodes);
    REQUIRE(s.nodes_per_depth == ref.nodes_per_depth);
    REQUIRE(sorted_types_csv(opt.out + "/types.csv") == want);
  }
}

TEST_CASE("batch and state messages round-trip and refuse other "
          "configurations", "[sched]") {
  const auto opt = d5_default(kTestOut + "/sched-wire");
  const SplitEnum::Walk w = SplitEnum::prepare(opt);
  const uint64_t h = SplitEnum::job_config_hash(w);
  SplitEnum::Pool pool(w, 2, 300);
  const JobResult r = pool.run({SplitEnum::root_job(w)}, 0.001, nullptr);
  REQUIRE(r.jobs > 0);

  SplitEnum::JobBatch batch;
  batch.jobs = r.rest;
  batch.seconds = 1.5;
  const std::vector<uint8_t> bb = SplitEnum::encode_jobs(batch, h);
  const SplitEnum::JobBatch db = SplitEnum::decode_jobs(bb, h);
  REQUIRE(db.jobs == batch.jobs);
  REQUIRE(db.seconds == batch.seconds);
  REQUIRE_THROWS(SplitEnum::decode_jobs(bb, h + 1));
  REQUIRE_THROWS(SplitEnum::decode_jobs(
      std::vector<uint8_t>(bb.begin(), bb.end() - 1), h));

  SplitEnum::JobState st;
  st.done = r;
  st.seconds = 42.25;
  const std::vector<uint8_t> sb = SplitEnum::encode_state(st, h);
  const SplitEnum::JobState ds = SplitEnum::decode_state(sb, h);
  REQUIRE(ds.seconds == st.seconds);
  REQUIRE(ds.done.jobs == r.jobs);
  REQUIRE(ds.done.rest == r.rest);
  REQUIRE(ds.done.nodes_per_depth == r.nodes_per_depth);
  REQUIRE(ds.done.types.size() == r.types.size());
  REQUIRE_THROWS(SplitEnum::decode_state(sb, h ^ 1));
  REQUIRE_THROWS(SplitEnum::decode_state(
      std::vector<uint8_t>(sb.begin(), sb.end() - 2), h));
  REQUIRE_THROWS(SplitEnum::decode_jobs(sb, h));
  REQUIRE_THROWS(SplitEnum::decode_state(bb, h));
}

TEST_CASE("schedulers refuse each other's directories", "[sched]") {
  auto chunks = d5_default(kTestOut + "/sched-mix-chunks");
  fs::remove_all(chunks.out);
  SplitEnum::run(chunks);
  auto onto_chunks = chunks;
  onto_chunks.scheduler_jobs = true;
  REQUIRE_THROWS_WITH(SplitEnum::run(onto_chunks),
                      Catch::Matchers::Contains("holds chunk files"));

  const auto jobs = sched_options(d5_default(""), kTestOut + "/sched-mix-jobs");
  SplitEnum::run(jobs);
  auto onto_jobs = jobs;
  onto_jobs.scheduler_jobs = false;
  REQUIRE_THROWS_WITH(SplitEnum::run(onto_jobs),
                      Catch::Matchers::Contains("holds the checkpoint"));

  auto other = jobs;
  other.symmetry = false;
  REQUIRE_THROWS_WITH(SplitEnum::run(other),
                      Catch::Matchers::Contains("different walk configuration"));
}

TEST_CASE("batch kernel classify reproduces the cpu types.csv", "[gpu]") {
  auto ref_opt = d5_default(kTestOut + "/gpu-ref");
  fs::remove_all(ref_opt.out);
  const auto ref = SplitEnum::run(ref_opt);
  const std::string want = sorted_types_csv(ref_opt.out + "/types.csv");
  for (bool refine : {false, true})
  for (int threads : {1, 4})
    for (uint64_t budget : {uint64_t{1}, uint64_t{100000}})
      for (int batch : {1, 7, 65536})
        for (uint64_t check : {uint64_t{0}, uint64_t{1}, uint64_t{3}}) {
          INFO("refine gpu " << refine << " threads " << threads << " budget "
                             << budget << " batch " << batch << " check "
                             << check);
          auto opt = sched_options(ref_opt, kTestOut + "/gpu-host");
          opt.refine_gpu = refine;
          opt.threads = threads;
          opt.budget = budget;
          opt.classify = SplitEnum::Classify::GpuHost;
          opt.gpu_batch = batch;
          opt.gpu_check = check;
          const SplitEnum::BatchStats before = SplitEnum::batch_stats();
          const auto s = SplitEnum::run(opt);
          const SplitEnum::BatchStats after = SplitEnum::batch_stats();
          REQUIRE_FALSE(s.stopped);
          REQUIRE(s.errors == 0);
          REQUIRE(s.total_nodes == ref.total_nodes);
          REQUIRE(s.nodes_per_depth == ref.nodes_per_depth);
          REQUIRE(sorted_types_csv(opt.out + "/types.csv") == want);
          REQUIRE(after.mismatches == before.mismatches);
          REQUIRE(after.fallbacks == before.fallbacks);
          REQUIRE(after.nodes > before.nodes);
          if (check == 1)
            REQUIRE(after.checked - before.checked + after.new_codes -
                        before.new_codes == after.nodes - before.nodes);
        }
}

TEST_CASE("batch kernel classify resumes and survives fake ranks", "[gpu]") {
  auto ref_opt = d5_default(kTestOut + "/gpu-resume-ref");
  fs::remove_all(ref_opt.out);
  const auto ref = SplitEnum::run(ref_opt);
  for (bool refine : {false, true}) {
    INFO("refine gpu " << refine);
    auto opt = sched_options(ref_opt, kTestOut + "/gpu-resume");
    opt.classify = SplitEnum::Classify::GpuHost;
    opt.refine_gpu = refine;
    opt.gpu_batch = 50;
    opt.budget = 200;
    int rounds = 0;
    SplitEnum::RunStats s;
    do {
      FakeRanks runner(3, 2, opt.budget, 4);
      s = SplitEnum::run(opt, &runner);
      ++rounds;
      REQUIRE(rounds < 100000);
    } while (s.stopped);
    REQUIRE(rounds > 1);
    REQUIRE(s.errors == 0);
    REQUIRE(s.nodes_per_depth == ref.nodes_per_depth);
    REQUIRE(sorted_types_csv(opt.out + "/types.csv") ==
            sorted_types_csv(ref_opt.out + "/types.csv"));
  }
}

TEST_CASE("batch kernel classify refusals", "[gpu]") {
  auto chunks = d5_default(kTestOut + "/gpu-refuse");
  fs::remove_all(chunks.out);
  chunks.classify = SplitEnum::Classify::GpuHost;
  REQUIRE_THROWS_WITH(SplitEnum::run(chunks),
                      Catch::Matchers::Contains("needs --scheduler jobs"));
  auto device = sched_options(chunks, kTestOut + "/gpu-refuse");
  device.classify = SplitEnum::Classify::Gpu;
  REQUIRE_THROWS_WITH(SplitEnum::run(device),
                      Catch::Matchers::Contains("no CUDA backend"));
  auto ref = d5_default(kTestOut + "/gpu-refuse-ref");
  fs::remove_all(ref.out);
  const auto want = SplitEnum::run(ref);
  auto cpu_refine = sched_options(chunks, kTestOut + "/gpu-refuse");
  cpu_refine.classify = SplitEnum::Classify::Cpu;
  cpu_refine.refine_gpu = true;
  const auto got = SplitEnum::run(cpu_refine);
  REQUIRE(got.total_nodes == want.total_nodes);
  REQUIRE(sorted_types_csv(cpu_refine.out + "/types.csv") ==
          sorted_types_csv(ref.out + "/types.csv"));
}

TEST_CASE("degree-6 batch kernel classify reproduces types.csv",
          "[.][d6][gpu]") {
  auto ref_opt = d5_default(kTestOut + "/gpu-d6-ref");
  ref_opt.delta = 6;
  fs::remove_all(ref_opt.out);
  const auto ref = SplitEnum::run(ref_opt);
  for (bool refine : {false, true}) {
    INFO("refine gpu " << refine);
    auto opt = sched_options(ref_opt, kTestOut + "/gpu-d6");
    opt.slice_seconds = 0.2;
    opt.checkpoint_seconds = 0.5;
    opt.classify = SplitEnum::Classify::GpuHost;
    opt.refine_gpu = refine;
    opt.gpu_check = 8;
    const auto s = SplitEnum::run(opt);
    REQUIRE(s.errors == 0);
    REQUIRE(s.total_nodes == ref.total_nodes);
    REQUIRE(s.nodes_per_depth == ref.nodes_per_depth);
    REQUIRE(sorted_types_csv(opt.out + "/types.csv") ==
            sorted_types_csv(ref_opt.out + "/types.csv"));
  }
}

namespace {
void wait_for_file(const std::string& path) {
  for (int i = 0; i < 6000 && !fs::exists(path); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  REQUIRE(fs::exists(path));
}

SplitEnum::Socket connect_master(const std::string& dir) {
  std::string host;
  int port = 0;
  REQUIRE(SplitEnum::read_addr(dir, host, port));
  SplitEnum::Socket sock = SplitEnum::connect_socket(host, port);
  REQUIRE(sock.is_open());
  return sock;
}

std::vector<uint8_t> hello(uint64_t h) {
  std::vector<uint8_t> b(8);
  for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>(h >> (8 * i));
  return b;
}
}

TEST_CASE("network master and workers reproduce the chunked types.csv",
          "[net]") {
  auto ref_opt = d5_default(kTestOut + "/net-ref");
  fs::remove_all(ref_opt.out);
  const auto ref = SplitEnum::run(ref_opt);
  auto opt = sched_options(ref_opt, kTestOut + "/net");
  opt.slice_seconds = 0.05;
  opt.budget = 300;
  const uint64_t hash = SplitEnum::job_config_hash(SplitEnum::prepare(opt));

  SplitEnum::TcpRunner runner(opt.out, 0);
  SplitEnum::RunStats s;
  std::thread master([&] {
    s = SplitEnum::run(opt, &runner);
    runner.finish();
  });
  wait_for_file(SplitEnum::addr_path(opt.out));

  {
    SplitEnum::Socket wrong = connect_master(opt.out);
    REQUIRE(wrong.send_frame(SplitEnum::kFrameHello, hello(hash ^ 1)));
    uint8_t tag = 0;
    std::vector<uint8_t> bytes;
    REQUIRE(wrong.recv_frame(tag, bytes));
    REQUIRE(tag == SplitEnum::kFrameRefuse);
  }
  {
    SplitEnum::Socket rogue = connect_master(opt.out);
    REQUIRE(rogue.send_frame(SplitEnum::kFrameHello, hello(hash)));
    uint8_t tag = 0;
    std::vector<uint8_t> bytes;
    REQUIRE(rogue.recv_frame(tag, bytes));
    REQUIRE(tag == SplitEnum::kFrameHello);
    REQUIRE(rogue.recv_frame(tag, bytes));
    REQUIRE(tag == SplitEnum::kFrameJobs);
    REQUIRE_FALSE(SplitEnum::decode_jobs(bytes, hash).jobs.empty());
  }

  std::vector<std::thread> workers;
  for (int k = 0; k < 2; ++k)
    workers.emplace_back([&, k] {
      auto w = opt;
      w.connect = opt.out;
      w.threads = 2;
      w.classify = k == 0 ? SplitEnum::Classify::Cpu
                          : SplitEnum::Classify::GpuHost;
      SplitEnum::run_worker(w);
    });
  master.join();
  for (std::thread& t : workers) t.join();

  REQUIRE_FALSE(s.stopped);
  REQUIRE(s.errors == 0);
  REQUIRE(s.total_nodes == ref.total_nodes);
  REQUIRE(s.nodes_per_depth == ref.nodes_per_depth);
  REQUIRE(sorted_types_csv(opt.out + "/types.csv") ==
          sorted_types_csv(ref_opt.out + "/types.csv"));
  REQUIRE_FALSE(fs::exists(SplitEnum::addr_path(opt.out)));
}

TEST_CASE("network worker leaves when the run is finished", "[net]") {
  auto opt = sched_options(d5_default(""), kTestOut + "/net-done");
  SplitEnum::run(opt);
  auto w = opt;
  w.connect = opt.out;
  w.connect_patience = 5;
  SplitEnum::run_worker(w);
  auto lost = opt;
  lost.connect = kTestOut + "/net-nowhere";
  lost.connect_patience = 0.5;
  fs::remove_all(lost.connect);
  REQUIRE_THROWS_WITH(SplitEnum::run_worker(lost),
                      Catch::Matchers::Contains("no master reachable"));
}

TEST_CASE("degree-6 job scheduler reproduces types.csv", "[.][d6][sched]") {
  auto ref_opt = d5_default(kTestOut + "/sched-d6-ref");
  ref_opt.delta = 6;
  fs::remove_all(ref_opt.out);
  const auto ref = SplitEnum::run(ref_opt);
  auto opt = sched_options(ref_opt, kTestOut + "/sched-d6");
  opt.slice_seconds = 0.2;
  opt.checkpoint_seconds = 0.5;
  const auto s = SplitEnum::run(opt);
  REQUIRE(s.total_nodes == ref.total_nodes);
  REQUIRE(s.nodes_per_depth == ref.nodes_per_depth);
  REQUIRE(sorted_types_csv(opt.out + "/types.csv") ==
          sorted_types_csv(ref_opt.out + "/types.csv"));
}
