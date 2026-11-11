#include "walk.h"

#include "gpuclassify.h"

#include <algorithm>
#include <stdexcept>

namespace SplitEnum {
ClassifyResult classify_sig(const SplitTable& table, SignMask sig,
                            const uint16_t* ids, size_t n) {
  ClassifyResult out;
  std::vector<Isotopy::Edge> edges;
  edges.reserve(2 * n);
  for (size_t k = 0; k < n; ++k) {
    const SplitRec& rec = table.recs[ids[k]];
    edges.push_back(rec.idx_a);
    if (rec.nsegs == 2) edges.push_back(rec.idx_b);
  }
  try {
    auto tris = Isotopy::refine(table.delta, edges);
    if (!tris) {
      out.viro = kErrType;
      out.err = true;
      return out;
    }
    Isotopy::Graph g(table.delta, unpack_signs(sig, table.npoints), *tris);
    out.viro = g.viro_notation();
    out.p = g.p_regions;
    out.n = g.n_regions;
    out.mcurve = g.is_mcurve();
  } catch (const std::exception&) {
    out.viro = kErrType;
    out.err = true;
  }
  return out;
}

ClassifyResult classify_lean(const SplitTable& table, const LeanRefine& lean,
                             SignMask sig, const uint64_t* forced,
                             const uint64_t* blocked) {
  ClassifyResult out;
  uint64_t acc[LeanRefine::kWords];
  std::copy(forced, forced + LeanRefine::kWords, acc);
  if (!lean.sweep(acc, blocked)) {
    out.viro = kErrType;
    out.err = true;
    return out;
  }
  uint8_t bytes[2 * 64 * LeanRefine::kWords];
  lean.edges(acc, bytes);
  std::vector<Isotopy::Edge> edges(lean.nedges());
  for (int e = 0; e < lean.nedges(); ++e)
    edges[e] = {bytes[2 * e], bytes[2 * e + 1]};
  try {
    Isotopy::Graph g(table.delta, unpack_signs(sig, table.npoints), edges);
    out.viro = g.viro_notation();
    out.p = g.p_regions;
    out.n = g.n_regions;
    out.mcurve = g.is_mcurve();
  } catch (const std::exception&) {
    out.viro = kErrType;
    out.err = true;
  }
  return out;
}

Walk prepare(const Options& opt) {
  Walk w;
  w.opt = &opt;
  w.table = build_splits(opt.delta, opt.family, opt.drop_same_side,
                         opt.drop_trivial, opt.drop_collinear);
  const int n = w.table.size();
  if (n >= 65536)
    throw std::runtime_error("more than 65535 splits: the witness's uint16 "
                             "split id no longer fits");
  if (n == 0) throw std::runtime_error("no splits for these options");
  if (opt.chunk_pairs < 1) throw std::runtime_error("--chunk-pairs must be >= 1");
  if (opt.shard_count < 1 || opt.shard_index < 0 ||
      opt.shard_index >= opt.shard_count)
    throw std::runtime_error("--shard needs 0 <= I < N");

  for (int i = 0; i < n; ++i) {
    const std::vector<bool> ref =
        Isotopy::signs_from_splits(opt.delta, {w.table.splits[i]});
    if (unpack_signs(w.table.harnack ^ w.table.recs[i].twist,
                     w.table.npoints) != ref)
      throw std::runtime_error("twist mask disagrees with signs_from_splits "
                               "for split " + std::to_string(i));
  }

  if (opt.symmetry) w.sym = std::make_unique<Symmetry>(w.table);

  if (!opt.count_only) {
    w.lean = std::make_unique<LeanRefine>(w.table);
    if (!w.lean->matches_library(w.table, nullptr, 0))
      throw std::runtime_error("lean refine disagrees with Isotopy::refine "
                               "on the empty collection");
    for (int i = 0; i < n; ++i) {
      const uint16_t id = static_cast<uint16_t>(i);
      if (!w.lean->matches_library(w.table, &id, 1))
        throw std::runtime_error("lean refine disagrees with Isotopy::refine "
                                 "for split " + std::to_string(i));
    }
  }

  w.compat = build_compat(w.table);
  if (w.sym && w.sym->words() != w.compat.words)
    throw std::runtime_error("symmetry/compat bitset word counts disagree");
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      if (w.compat.get(i, j))
        w.pairs.emplace_back(static_cast<uint16_t>(i),
                             static_cast<uint16_t>(j));
  w.npairs = w.pairs.size();
  w.chunk_count =
      opt.max_depth == 1
          ? 0
          : static_cast<int>((w.npairs + opt.chunk_pairs - 1) / opt.chunk_pairs);

  w.base.delta = opt.delta;
  w.base.family = static_cast<int>(opt.family);
  w.base.drop_same_side = opt.drop_same_side;
  w.base.count_only = opt.count_only;
  w.base.symmetry = opt.symmetry;
  w.base.max_depth = opt.max_depth;
  w.base.nsplits = n;
  w.base.table_hash = table_hash(w.table, opt.family, opt.drop_same_side);
  w.base.chunk_count = static_cast<uint64_t>(w.chunk_count);
  w.base.npairs = w.npairs;
  return w;
}

void Ctx::init(const Walk& w) {
  const Options& opt = *w.opt;
  table = &w.table;
  compat = &w.compat;
  sym = w.sym.get();
  lean = w.lean.get();
  if (lean) {
    lean_levels.assign(
        static_cast<size_t>(lean->nedges() + 2) * 2 * LeanRefine::kWords, 0);
    lean_ids.assign(lean->nedges() + 2, 0);
    lean_top = 0;
  }
  words = w.compat.words;
  max_depth = opt.max_depth;
  count_only = opt.count_only;
  sym_order = w.sym ? static_cast<uint32_t>(w.sym->order()) : 1;
  stack.reserve(w.table.size() + 2);
  scratch.assign(static_cast<size_t>(w.table.size() + 2) * w.compat.words, 0);
  sym_scratch.assign(w.table.size() + 2, 0);
}

void Ctx::reset() {
  stack.clear();
  sig = table->harnack;
  per_depth.clear();
  chunk_types = TypeTable();
  classified = 0;
  chunk_errs = 0;
}

void Ctx::emit(int depth, uint32_t weight) {
  if (static_cast<size_t>(depth) >= per_depth.size())
    per_depth.resize(depth + 1, 0);
  per_depth[depth] += weight;
  if (count_only) return;
  classify_node(depth, weight);
}

const uint64_t* Ctx::lean_seed() {
  constexpr int kW = LeanRefine::kWords;
  const size_t d = stack.size();
  const size_t keep = std::min(lean_top, d);
  size_t k = 0;
  while (k < keep && lean_ids[k] == stack[k]) ++k;
  for (; k < d; ++k) {
    const uint64_t* src = lean_levels.data() + k * 2 * kW;
    uint64_t* dst = lean_levels.data() + (k + 1) * 2 * kW;
    const uint64_t* f = lean->forced_row(stack[k]);
    const uint64_t* b = lean->blocked_row(stack[k]);
    for (int w = 0; w < kW; ++w) {
      dst[w] = src[w] | f[w];
      dst[kW + w] = src[kW + w] | b[w];
    }
    lean_ids[k] = stack[k];
  }
  lean_top = d;
  return lean_levels.data() + d * 2 * kW;
}

void Ctx::classify_node(int depth, uint32_t weight) {
  if (batch && batch->refines()) {
    ++classify_calls;
    ++classified;
    batch->add_ids(sig,
                   profile_of(*table, stack.data(), static_cast<int>(stack.size())),
                   weight, depth, stack.data(), stack.size());
    if (batch->full()) batch->rotate(chunk_types, chunk_errs);
    return;
  }
  const uint64_t* seed = lean_seed();
  if (batch) {
    ++classify_calls;
    ++classified;
    const Profile profile =
        profile_of(*table, stack.data(), static_cast<int>(stack.size()));
    if (!batch->add(sig, seed, seed + LeanRefine::kWords, profile, weight,
                    depth, stack.data(), stack.size())) {
      ++chunk_errs;
      chunk_types.add(kErrType, -1, -1, false, profile, weight, depth,
                      stack.data(), stack.size());
    } else if (batch->full()) {
      batch->rotate(chunk_types, chunk_errs);
    }
    return;
  }
  const ClassifyResult res =
      classify_lean(*table, *lean, sig, seed, seed + LeanRefine::kWords);
  ++classify_calls;
  ++classified;
  if (res.err) ++chunk_errs;
  chunk_types.add(res.viro, res.p, res.n, res.mcurve,
                  profile_of(*table, stack.data(), static_cast<int>(stack.size())),
                  weight, depth, stack.data(), stack.size());
}

inline void Ctx::visit(const uint64_t* cand, int depth, int v) {
  push(v);
  unsigned add = 0;
  for (int i = 0; i < ntargets; ++i)
    add |= (v == targets[i] ? 1u : 0u) << i;
  active ^= add;
  uint32_t weight = 1;
  if (canonical_masked(weight)) {
    emit(depth, weight);
    if (max_depth == 0 || depth < max_depth) {
      uint64_t* nc = level(depth);
      if (and_above(nc, cand, compat->row(v), words, v)) dfs(nc, depth + 1);
    }
  }
  active ^= add;
  pop(v);
}

void Ctx::dfs(const uint64_t* cand, int depth) {
  if (depth <= 6 && abort && abort->load(std::memory_order_relaxed)) {
    if (!rest) return;
    budget = 0;
  }
  for (int w = 0; w < words; ++w) {
    uint64_t x = cand[w];
    while (x) {
      const int v = (w << 6) + __builtin_ctzll(x);
      x &= x - 1;
      if (budget == 0) [[unlikely]] {
        yield(depth, v);
        return;
      }
      --budget;
      visit(cand, depth, v);
    }
  }
}

void Ctx::dfs_below(const uint64_t* cand, int depth, int hi) {
  for (int w = 0; w < words; ++w) {
    uint64_t x = cand[w];
    while (x) {
      const int v = (w << 6) + __builtin_ctzll(x);
      if (v >= hi) return;
      x &= x - 1;
      if (budget == 0) {
        yield(depth, v);
        return;
      }
      --budget;
      visit(cand, depth, v);
    }
  }
}

void Ctx::yield(int depth, int v) {
  Job j;
  j.prefix.assign(stack.begin(), stack.end());
  j.lo = static_cast<uint16_t>(v);
  j.hi = depth == job_depth ? job_hi : static_cast<uint16_t>(table->size());
  rest->push_back(std::move(j));
}

bool Ctx::enter_pair(const Walk& w, int u, int v) {
  uint64_t* nc = level(2);
  if (!w.sym) return and_above(nc, w.compat.row(u), w.compat.row(v), words, v);
  const auto& elems = w.sym->elems();
  ntargets = static_cast<int>(elems.size());
  active = 0;
  for (int e = 0; e < ntargets; ++e) {
    targets[e] = elems[e].inv_perm[u];
    if (targets[e] == u || targets[e] == v) active |= 1u << e;
  }
  return and_above_masked(nc, w.compat.row(u), w.compat.row(v),
                          w.sym->allowed_row(u), words, v);
}

namespace {
bool clip(uint64_t* dst, int words, int lo, int hi) {
  uint64_t any = 0;
  for (int wd = 0; wd < words; ++wd) {
    const int base = wd << 6;
    uint64_t m = ~uint64_t{0};
    if (lo >= base + 64) m = 0;
    else if (lo > base) m &= ~uint64_t{0} << (lo - base);
    if (hi <= base) m = 0;
    else if (hi < base + 64) m &= ~uint64_t{0} >> (base + 64 - hi);
    dst[wd] &= m;
    any |= dst[wd];
  }
  return any != 0;
}

bool prefix_candidates(const Walk& w, const uint16_t* prefix, int k, int lo,
                       int hi, uint64_t* dst) {
  const int words = w.compat.words;
  hi = std::min(hi, w.table.size());
  if (k == 0) {
    std::fill(dst, dst + words, ~uint64_t{0});
    return clip(dst, words, lo, hi);
  }
  const int last = prefix[k - 1];
  if (k == 1) {
    std::copy(w.compat.row(last), w.compat.row(last) + words, dst);
    return clip(dst, words, std::max(lo, last + 1), hi);
  }
  const int p0 = prefix[0], p1 = prefix[1];
  bool live = w.sym ? and_above_masked(dst, w.compat.row(p0), w.compat.row(p1),
                                       w.sym->allowed_row(p0), words, p1)
                    : and_above(dst, w.compat.row(p0), w.compat.row(p1), words,
                                p1);
  for (int i = 2; i < k && live; ++i)
    live = and_above(dst, dst, w.compat.row(prefix[i]), words, prefix[i]);
  return live && clip(dst, words, std::max(lo, last + 1), hi);
}
}

void Ctx::pair_children(const Walk& w, int u, int lo, int hi) {
  const uint16_t pu = static_cast<uint16_t>(u);
  uint64_t* cand = level(1);
  if (!prefix_candidates(w, &pu, 1, lo, hi, cand)) return;
  for (int wd = 0; wd < words; ++wd) {
    uint64_t x = cand[wd];
    while (x) {
      const int v = (wd << 6) + __builtin_ctzll(x);
      x &= x - 1;
      if (budget == 0) {
        yield(2, v);
        return;
      }
      --budget;
      push(v);
      uint32_t weight = 1;
      if (canonical(weight)) {
        emit(2, weight);
        if ((max_depth == 0 || 2 < max_depth) && enter_pair(w, u, v))
          dfs(level(2), 3);
      }
      pop(v);
    }
  }
}

void run_job(Ctx& c, const Walk& w, const Job& job, uint64_t budget,
             std::vector<Job>& rest) {
  const int k = static_cast<int>(job.prefix.size());
  c.budget = budget;
  c.rest = &rest;
  c.job_depth = k + 1;
  c.job_hi = job.hi;
  if (c.abort && c.abort->load(std::memory_order_relaxed)) c.budget = 0;
  for (uint16_t p : job.prefix) c.push(p);
  bool open = true;
  if (job.root) {
    if (c.budget == 0) {
      rest.push_back(job);
      open = false;
    } else {
      --c.budget;
      c.emit(0, 1);
    }
  }
  if (open && (c.max_depth == 0 || k < c.max_depth)) {
    if (k == 0) {
      for (int v = job.lo; v < job.hi; ++v) {
        if (c.budget == 0) {
          c.yield(1, v);
          break;
        }
        --c.budget;
        c.push(v);
        uint32_t weight = 1;
        if (c.canonical(weight)) {
          c.emit(1, weight);
          if (c.max_depth != 1) c.pair_children(w, v, v + 1, w.table.size());
        }
        c.pop(v);
      }
    } else if (k == 1) {
      c.pair_children(w, job.prefix[0], job.lo, job.hi);
    } else {
      if (w.sym) {
        const auto& elems = w.sym->elems();
        c.ntargets = static_cast<int>(elems.size());
        c.active = 0;
        for (int e = 0; e < c.ntargets; ++e) {
          c.targets[e] = elems[e].inv_perm[job.prefix[0]];
          if (std::binary_search(job.prefix.begin(), job.prefix.end(),
                                 c.targets[e]))
            c.active |= 1u << e;
        }
      }
      uint64_t* nc = c.level(k);
      if (prefix_candidates(w, job.prefix.data(), k, job.lo, w.table.size(),
                            nc))
        c.dfs_below(nc, k + 1, job.hi);
    }
  }
  for (int i = k - 1; i >= 0; --i) c.pop(job.prefix[i]);
  c.budget = UINT64_MAX;
  c.rest = nullptr;
}

std::vector<uint16_t> job_candidates(const Walk& w, const Job& job) {
  std::vector<uint64_t> bits(w.compat.words);
  std::vector<uint16_t> out;
  if (!prefix_candidates(w, job.prefix.data(),
                         static_cast<int>(job.prefix.size()), job.lo, job.hi,
                         bits.data()))
    return out;
  for (int wd = 0; wd < w.compat.words; ++wd)
    for (uint64_t x = bits[wd]; x; x &= x - 1)
      out.push_back(static_cast<uint16_t>((wd << 6) + __builtin_ctzll(x)));
  return out;
}

void walk_nodes(Ctx& c, const Walk& w, int k) {
  c.reset();
  if (k == kShallowChunk) {
    c.emit(0, 1);
    for (int v = 0; v < w.table.size(); ++v) {
      c.push(v);
      uint32_t weight = 1;
      if (c.canonical(weight)) c.emit(1, weight);
      c.pop(v);
    }
    return;
  }
  const uint64_t lo = static_cast<uint64_t>(k) * w.opt->chunk_pairs;
  const uint64_t hi = std::min<uint64_t>(w.npairs, lo + w.opt->chunk_pairs);
  for (uint64_t i = lo; i < hi; ++i) {
    const int u = w.pairs[i].first, v = w.pairs[i].second;
    c.push(u);
    c.push(v);
    uint32_t weight = 1;
    if (c.canonical(weight)) {
      c.emit(2, weight);
      if ((c.max_depth == 0 || 2 < c.max_depth) && c.enter_pair(w, u, v))
        c.dfs(c.level(2), 3);
    }
    c.pop(v);
    c.pop(u);
  }
}
}
