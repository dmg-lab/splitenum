#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "gpuclassify.h"
#include "jobs.h"
#include "pool.h"
#include "walk.h"

namespace {
using namespace SplitEnum;
using Clock = std::chrono::steady_clock;

constexpr double kWindow = 10.0;

double since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::vector<uint64_t> spread_order(uint64_t n) {
  uint64_t step = static_cast<uint64_t>(0.6180339887 * static_cast<double>(n));
  step = std::max<uint64_t>(step, 1);
  while (std::gcd(step, n) != 1) ++step;
  std::vector<uint64_t> order(n);
  for (uint64_t i = 0; i < n; ++i) order[i] = (i * step) % n;
  return order;
}

struct Totals {
  uint64_t nodes = 0;
  uint64_t subtrees = 0;
  uint64_t classify = 0;
  double seconds = 0;
};

Totals run_walk(const Options& base, bool count_only, double seconds,
                int nthreads) {
  Options opt = base;
  opt.count_only = count_only;
  opt.chunk_pairs = 1;
  const Walk w = prepare(opt);
  const std::vector<uint64_t> order = spread_order(w.npairs);
  std::atomic<uint64_t> cursor{0};
  std::atomic<bool> stop{false};
  std::vector<Totals> per(nthreads);
  const auto t0 = Clock::now();
  const auto work = [&](int t) {
    Ctx c;
    c.init(w);
    c.abort = &stop;
    Totals& r = per[t];
    while (!stop.load()) {
      const uint64_t i = cursor.fetch_add(1) % w.npairs;
      walk_nodes(c, w, static_cast<int>(order[i]));
      for (uint64_t v : c.per_depth) r.nodes += v;
      ++r.subtrees;
    }
    r.classify = c.classify_calls;
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < nthreads; ++t) pool.emplace_back(work, t);
  while (since(t0) < seconds)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  stop = true;
  for (auto& th : pool) th.join();
  Totals sum;
  sum.seconds = since(t0);
  for (const Totals& r : per) {
    sum.nodes += r.nodes;
    sum.subtrees += r.subtrees;
    sum.classify += r.classify;
  }
  return sum;
}

struct JobTotals {
  uint64_t nodes = 0;
  uint64_t jobs = 0;
  uint64_t drains = 0;
  uint64_t classify = 0;
  double seconds = 0;
};

JobTotals run_jobs(const Options& base, bool count_only, uint64_t budget,
                   double seconds, int nthreads, bool feed = false) {
  Options opt = base;
  opt.count_only = count_only;
  if (feed) {
    opt.classify = Classify::Feed;
    opt.gpu_check = 0;
  }
  const Walk w = prepare(opt);
  Pool pool(w, nthreads, budget);
  JobTotals t;
  const auto t0 = Clock::now();
  while (since(t0) < seconds) {
    const JobResult r = pool.run({root_job(w)}, seconds - since(t0), nullptr);
    t.nodes += r.nodes();
    t.classify += r.classify_calls;
    t.jobs += r.jobs;
    t.drains += r.finished();
  }
  t.seconds = since(t0);
  return t;
}

double per_window(uint64_t n, double seconds) {
  return static_cast<double>(n) * kWindow / seconds;
}

template <typename F>
double micro(double seconds, size_t n, F&& op) {
  uint64_t ops = 0;
  const auto t0 = Clock::now();
  double el = 0;
  do {
    for (size_t i = 0; i < n; ++i) op(i);
    ops += n;
    el = since(t0);
  } while (el < seconds);
  return per_window(ops, el);
}

struct Sample {
  SignMask sig = 0;
  std::vector<uint16_t> ids;
};

std::vector<Sample> sample_collections(const Walk& w, size_t count,
                                       uint64_t seed) {
  std::mt19937_64 rng(seed);
  const int n = w.table.size();
  std::vector<Sample> out;
  std::vector<uint64_t> cand(w.compat.words);
  while (out.size() < count) {
    Sample s;
    s.sig = w.table.harnack;
    std::fill(cand.begin(), cand.end(), ~uint64_t{0});
    const int size = static_cast<int>(rng() % 13);
    int last = -1;
    for (int k = 0; k < size; ++k) {
      std::vector<uint16_t> ok;
      for (int v = last + 1; v < n; ++v)
        if (cand[v >> 6] >> (v & 63) & 1) ok.push_back(static_cast<uint16_t>(v));
      if (ok.empty()) break;
      const uint16_t v = ok[rng() % ok.size()];
      s.ids.push_back(v);
      s.sig ^= w.table.recs[v].twist;
      for (int wd = 0; wd < w.compat.words; ++wd) cand[wd] &= w.compat.row(v)[wd];
      last = v;
    }
    out.push_back(std::move(s));
  }
  return out;
}

void run_micro(const Options& base, double seconds) {
  Options opt = base;
  opt.count_only = false;
  const Walk w = prepare(opt);
  const std::vector<Sample> recs = sample_collections(w, 20000, 1);
  const size_t n = recs.size();
  std::vector<uint16_t> scratch(w.table.size() + 2);
  uint64_t sink = 0;

  std::printf("micro   1 thread, %zu sampled collections, rates per %.0fs\n", n,
              kWindow);
  const auto show = [](const char* name, double rate) {
    std::printf("  %-24s %14.4g\n", name, rate);
  };
  if (w.sym) {
    show("canonical_stack", micro(seconds, n, [&](size_t i) {
           uint32_t wt = 0;
           sink += w.sym->canonical_stack(recs[i].ids.data(),
                                          static_cast<int>(recs[i].ids.size()),
                                          scratch.data(), wt) + wt;
         }));
  }
  uint64_t acc[LeanRefine::kWords];
  std::vector<uint8_t> bytes(2 * static_cast<size_t>(w.lean->nedges()));
  show("lean_refine", micro(seconds, n, [&](size_t i) {
         w.lean->refine(recs[i].ids.data(), static_cast<int>(recs[i].ids.size()),
                        acc);
         w.lean->edges(acc, bytes.data());
         sink += bytes[1];
       }));
  show("classify_lean", micro(seconds, std::min<size_t>(n, 2048), [&](size_t i) {
         uint64_t forced[LeanRefine::kWords] = {};
         uint64_t blocked[LeanRefine::kWords] = {};
         for (uint16_t id : recs[i].ids)
           for (int wd = 0; wd < LeanRefine::kWords; ++wd) {
             forced[wd] |= w.lean->forced_row(id)[wd];
             blocked[wd] |= w.lean->blocked_row(id)[wd];
           }
         sink += classify_lean(w.table, *w.lean, recs[i].sig, forced, blocked).p;
       }));
  show("classify_sig", micro(seconds, std::min<size_t>(n, 256), [&](size_t i) {
         sink += classify_sig(w.table, recs[i].sig, recs[i].ids.data(),
                              recs[i].ids.size()).p;
       }));
  if (sink == 0) std::printf("  (no work sampled)\n");
}

bool parse_bool(const char* v, const char* yes, const char* no, bool& out) {
  if (!std::strcmp(v, yes)) return out = true, true;
  if (!std::strcmp(v, no)) return out = false, true;
  return false;
}
}

int main(int argc, char** argv) {
  Options opt;
  opt.delta = 6;
  double seconds = kWindow;
  double micro_seconds = 1.0;
  int nthreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  std::string mode = "all";
  uint64_t budget = 1000000;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) {
      std::fprintf(stderr, "bench: %s needs a value\n", a.c_str());
      return 2;
    }
    const char* v = argv[++i];
    bool ok = true;
    if (a == "--delta") opt.delta = std::atoi(v);
    else if (a == "--seconds") seconds = std::atof(v);
    else if (a == "--micro-seconds") micro_seconds = std::atof(v);
    else if (a == "--threads") nthreads = std::max(1, std::atoi(v));
    else if (a == "--mode") mode = v;
    else if (a == "--budget") budget = std::max<uint64_t>(1, std::strtoull(v, nullptr, 10));
    else if (a == "--symmetry") ok = parse_bool(v, "on", "off", opt.symmetry);
    else if (a == "--trivial") ok = parse_bool(v, "drop", "keep", opt.drop_trivial);
    else if (a == "--collinear") ok = parse_bool(v, "drop", "keep", opt.drop_collinear);
    else if (a == "--refine") ok = parse_bool(v, "gpu", "cpu", opt.refine_gpu);
    else ok = false;
    if (!ok) {
      std::fprintf(stderr,
                   "usage: bench [--delta N] [--seconds S] [--micro-seconds S] "
                   "[--threads T] [--mode count|full|micro|jobs|jobs-full|feed|all] "
                   "[--budget N] "
                   "[--symmetry on|off] [--trivial drop|keep] "
                   "[--collinear drop|keep] "
                   "[--refine cpu|gpu]\n");
      return 2;
    }
  }
  if (opt.delta >= 8) {
    std::fprintf(stderr, "bench: delta %d is not a benchmark degree, use 6 or 7\n",
                 opt.delta);
    return 2;
  }

  try {
    const Walk w = [&] {
      Options o = opt;
      o.count_only = true;
      return prepare(o);
    }();
    std::printf("bench   delta %d, %d splits, %llu pairs, symmetry %s, %d threads\n",
                opt.delta, w.table.size(),
                static_cast<unsigned long long>(w.npairs),
                opt.symmetry ? "on" : "off", nthreads);
    const bool all = mode == "all";
    if (all || mode == "count") {
      const Totals t = run_walk(opt, true, seconds, nthreads);
      std::printf("count   %14.4g nodes/10s  %12.4g nodes/s  %llu subtrees in %.2fs\n",
                  per_window(t.nodes, t.seconds), t.nodes / t.seconds,
                  static_cast<unsigned long long>(t.subtrees), t.seconds);
    }
    if (all || mode == "full") {
      const Totals t = run_walk(opt, false, seconds, nthreads);
      std::printf("full    %14.4g nodes/10s  %12.4g classify/10s  "
                  "%llu subtrees in %.2fs\n",
                  per_window(t.nodes, t.seconds),
                  per_window(t.classify, t.seconds),
                  static_cast<unsigned long long>(t.subtrees), t.seconds);
    }
    if (mode == "jobs" || mode == "jobs-full") {
      const JobTotals t =
          run_jobs(opt, mode == "jobs", budget, seconds, nthreads);
      std::printf("%-7s budget %llu  %12.4g nodes/s  %10.4g jobs/s  "
                  "%12.4g classify/10s  %llu drains in %.2fs\n",
                  mode.c_str(), static_cast<unsigned long long>(budget),
                  t.nodes / t.seconds, t.jobs / t.seconds,
                  per_window(t.classify, t.seconds),
                  static_cast<unsigned long long>(t.drains), t.seconds);
    }
    if (mode == "feed") {
      const BatchStats b0 = batch_stats();
      const JobTotals t = run_jobs(opt, false, budget, seconds, nthreads, true);
      const BatchStats b1 = batch_stats();
      const double nodes = static_cast<double>(t.nodes);
      const double busy = t.seconds * nthreads;
      const double run = (b1.run_ns - b0.run_ns) * 1e-9;
      const double resolve = (b1.resolve_ns - b0.resolve_ns) * 1e-9;
      std::printf("feed    budget %llu  %12.4g nodes/s  %8.4g nodes/s/thread  "
                  "%12.4g representatives/s\n"
                  "        per node: total %.4f us, run %.4f us, resolve %.4f us, "
                  "walk+append %.4f us\n",
                  static_cast<unsigned long long>(budget), nodes / t.seconds,
                  nodes / busy, t.classify / t.seconds, busy / nodes * 1e6, run / nodes * 1e6,
                  resolve / nodes * 1e6, (busy - run - resolve) / nodes * 1e6);
    }
    if (all || mode == "micro") run_micro(opt, micro_seconds);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench: %s\n", e.what());
    return 1;
  }
  return 0;
}
