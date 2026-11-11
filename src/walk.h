#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "chunkio.h"
#include "compat.h"
#include "enumerate.h"
#include "refine.h"
#include "splits.h"
#include "store.h"
#include "symmetry.h"

namespace SplitEnum {
inline constexpr const char* kErrType = "#ERR";

class BatchClassifier;

struct ClassifyResult {
  std::string viro;
  int p = -1;
  int n = -1;
  bool mcurve = false;
  bool err = false;
};

ClassifyResult classify_sig(const SplitTable& table, SignMask sig,
                            const uint16_t* ids, size_t n);

ClassifyResult classify_lean(const SplitTable& table, const LeanRefine& lean,
                             SignMask sig, const uint64_t* forced,
                             const uint64_t* blocked);

using PairList = std::vector<std::pair<uint16_t, uint16_t>>;

struct Job {
  std::vector<uint16_t> prefix;
  uint16_t lo = 0;
  uint16_t hi = 0;
  bool root = false;

  bool operator==(const Job&) const = default;
};

struct Walk {
  const Options* opt = nullptr;
  SplitTable table;
  Compat compat;
  std::unique_ptr<Symmetry> sym;
  std::unique_ptr<LeanRefine> lean;
  PairList pairs;
  uint64_t npairs = 0;
  int chunk_count = 0;
  ChunkHeader base;
};

Walk prepare(const Options& opt);

struct Ctx {
  const SplitTable* table = nullptr;
  const Compat* compat = nullptr;
  const Symmetry* sym = nullptr;
  const LeanRefine* lean = nullptr;
  std::shared_ptr<BatchClassifier> batch;
  bool batch_defer = false;
  int words = 0;
  int max_depth = 0;
  bool count_only = false;
  const std::atomic<bool>* abort = nullptr;

  uint64_t budget = UINT64_MAX;
  std::vector<Job>* rest = nullptr;
  int job_depth = 0;
  uint16_t job_hi = 0;

  std::vector<uint16_t> stack;
  std::vector<uint64_t> scratch;
  std::vector<uint16_t> sym_scratch;
  std::vector<uint64_t> lean_levels;
  std::vector<uint16_t> lean_ids;
  size_t lean_top = 0;
  SignMask sig = 0;

  uint32_t sym_order = 1;
  unsigned active = 0;
  int ntargets = 0;
  uint16_t targets[5] = {0, 0, 0, 0, 0};

  std::vector<uint64_t> per_depth;

  TypeTable chunk_types;
  uint64_t classified = 0;
  uint64_t chunk_errs = 0;
  uint64_t classify_calls = 0;

  void init(const Walk& w);
  void reset();
  void emit(int depth, uint32_t weight);
  void classify_node(int depth, uint32_t weight);
  const uint64_t* lean_seed();
  void dfs(const uint64_t* cand, int depth);
  void dfs_below(const uint64_t* cand, int depth, int hi);
  [[gnu::always_inline]] inline void visit(const uint64_t* cand, int depth,
                                           int v);
  [[gnu::cold, gnu::noinline]] void yield(int depth, int v);
  bool enter_pair(const Walk& w, int u, int v);
  void pair_children(const Walk& w, int u, int lo, int hi);

  uint64_t* level(int depth) {
    return scratch.data() + static_cast<size_t>(depth) * words;
  }

  void push(int v) {
    stack.push_back(static_cast<uint16_t>(v));
    sig ^= table->recs[v].twist;
  }

  void pop(int v) {
    sig ^= table->recs[v].twist;
    stack.pop_back();
  }

  bool canonical(uint32_t& weight) {
    return !sym || sym->canonical_stack(stack.data(),
                                        static_cast<int>(stack.size()),
                                        sym_scratch.data(), weight);
  }

  bool canonical_masked(uint32_t& weight) {
    if (!sym) return true;
    if (!active) {
      weight = sym_order;
      return true;
    }
    return sym->canonical_stack_active(stack.data(),
                                       static_cast<int>(stack.size()),
                                       sym_scratch.data(), active, weight);
  }
};

void walk_nodes(Ctx& c, const Walk& w, int k);

void run_job(Ctx& c, const Walk& w, const Job& job, uint64_t budget,
             std::vector<Job>& rest);

std::vector<uint16_t> job_candidates(const Walk& w, const Job& job);
}
