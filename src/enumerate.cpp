#include "enumerate.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <thread>

#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "master.h"
#include "pool.h"
#include "walk.h"

namespace fs = std::filesystem;

namespace SplitEnum {
namespace {
std::atomic<bool>& g_stop = stop_flag();

std::string chunk_path(const std::string& out, int k) {
  if (k == kShallowChunk) return out + "/chunks/walk-shallow.bin";
  char buf[32];
  std::snprintf(buf, sizeof buf, "/chunks/walk-c%06d.bin", k);
  return out + buf;
}

void check_existing_chunks(const Walk& w, const std::string& out) {
  const std::string dir = out + "/chunks";
  if (!fs::exists(dir)) return;
  for (const auto& e : fs::directory_iterator(dir)) {
    const std::string name = e.path().filename().string();
    if (name.rfind("walk-", 0) != 0 || e.path().extension() != ".bin")
      continue;
    const ChunkHeader h = read_chunk_header(e.path().string());
    if (!h.config_matches(w.base))
      throw std::runtime_error(
          e.path().string() +
          " was walked with a different configuration (degree, family, "
          "same-side, trivial, collinear, symmetry, max-depth, "
          "count-only or chunk-pairs); use a fresh --out directory");
  }
}

void walk_chunk(Ctx& c, const Walk& w, int k) {
  const auto t0 = std::chrono::steady_clock::now();
  walk_nodes(c, w, k);

  ChunkHeader h = w.base;
  h.chunk_index = k;
  if (k != kShallowChunk) {
    h.pair_lo = static_cast<uint64_t>(k) * w.opt->chunk_pairs;
    h.pair_hi = std::min<uint64_t>(w.npairs, h.pair_lo + w.opt->chunk_pairs);
  }
  h.nodes_per_depth = c.per_depth;
  h.nclassified = c.classified;
  h.nerr = c.chunk_errs;
  h.nrecords = c.chunk_types.types().size();
  h.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            t0).count();
  write_chunk(chunk_path(w.opt->out, k), h, c.chunk_types.types());
}

std::vector<int> missing_chunks(const Walk& w, bool own_shard) {
  const Options& opt = *w.opt;
  const auto mine = [&](int k) {
    return !own_shard || (k == kShallowChunk ? opt.shard_index == 0
                                             : k % opt.shard_count ==
                                                   opt.shard_index);
  };
  std::vector<int> todo;
  for (int k = kShallowChunk; k < w.chunk_count; ++k)
    if (mine(k) && !fs::exists(chunk_path(opt.out, k))) todo.push_back(k);
  return todo;
}

void walk_phase(const Walk& w, int nthreads, RunStats& s) {
  const Options& opt = *w.opt;
  const std::vector<int> todo = missing_chunks(w, true);
  uint64_t shard_chunks = w.chunk_count / opt.shard_count;
  shard_chunks += opt.shard_index <
                  static_cast<int>(w.chunk_count % opt.shard_count);
  if (opt.shard_index == 0) ++shard_chunks;
  s.chunks_total = shard_chunks;

  std::vector<Ctx> ctxs(nthreads);
  for (Ctx& c : ctxs) c.init(w);

  std::atomic<uint64_t> done{0};
  std::atomic<uint64_t> failed{0};
  const long long jobs = static_cast<long long>(todo.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (long long i = 0; i < jobs; ++i) {
    if (g_stop.load(std::memory_order_relaxed)) continue;
#ifdef _OPENMP
    Ctx& c = ctxs[omp_get_thread_num()];
#else
    Ctx& c = ctxs[0];
#endif
    try {
      walk_chunk(c, w, todo[i]);
    } catch (const std::exception& e) {
      ++failed;
      std::fprintf(stderr, "\nchunk %d failed: %s -- left missing, a rerun "
                   "retries it\n", todo[i], e.what());
    }
    const uint64_t d = ++done;
    if (d % 16 == 0 || d == static_cast<uint64_t>(jobs))
      std::fprintf(stderr, "\rwalk: chunks %llu/%llu",
                   static_cast<unsigned long long>(d),
                   static_cast<unsigned long long>(jobs));
  }
  if (jobs) std::fprintf(stderr, "\n");
  if (const uint64_t f = failed.load())
    std::fprintf(stderr, "walk: %llu chunks failed and are missing; rerun "
                 "the same command to retry them\n",
                 static_cast<unsigned long long>(f));

  s.chunks_done = done.load();
  s.stopped = g_stop.load();
}

void aggregate_walk_stats(const Walk& w, RunStats& s) {
  for (int k = kShallowChunk; k < w.chunk_count; ++k) {
    const ChunkHeader h = read_chunk_header(chunk_path(w.opt->out, k));
    for (size_t d = 0; d < h.nodes_per_depth.size(); ++d)
      s.add_depth(static_cast<int>(d), h.nodes_per_depth[d]);
  }
  s.total_nodes = 0;
  for (uint64_t k : s.nodes_per_depth) s.total_nodes += k;
  s.max_depth_reached =
      s.nodes_per_depth.empty()
          ? 0
          : static_cast<int>(s.nodes_per_depth.size()) - 1;
}

void require_complete_walk(const Walk& w) {
  const std::vector<int> missing = missing_chunks(w, false);
  if (missing.empty()) return;
  std::string what = "classify needs the complete walk but " +
                     std::to_string(missing.size()) +
                     " chunks are missing, first ids:";
  for (size_t i = 0; i < missing.size() && i < 8; ++i)
    what += ' ' + std::to_string(missing[i]);
  what += "; rerun the walk (all shards) first";
  throw std::runtime_error(what);
}

void classify_phase(const Walk& w, RunStats& s) {
  const Options& opt = *w.opt;
  require_complete_walk(w);
  TypeTable types;
  for (int k = kShallowChunk; k < w.chunk_count; ++k) {
    ChunkHeader h;
    for (const TypeInfo& t : read_chunk(chunk_path(opt.out, k), &h))
      types.add(t.viro, t.p, t.n, t.mcurve, t.profile, t.count, t.first_depth,
                t.witness.data(), t.witness.size());
    s.classified += h.nclassified;
    s.errors += h.nerr;
  }
  types.write_csv(opt.out + "/types.csv", opt.delta);
}

bool has_chunk_files(const std::string& out) {
  const std::string dir = out + "/chunks";
  if (!fs::exists(dir)) return false;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.path().filename().string().rfind("walk-", 0) == 0) return true;
  return false;
}

std::string state_path(const std::string& out) {
  return out + "/jobs/state.bin";
}

void jobs_run(const Walk& w, int nthreads, SliceRunner* remote, RunStats& s) {
  const Options& opt = *w.opt;
  if (opt.phase != Phase::All || opt.shard_count != 1)
    throw std::runtime_error("--scheduler jobs has no phases and no shards; "
                             "drop --phase and --shard");
  if (opt.classify != Classify::Cpu && !opt.count_only && opt.gpu_batch < 1)
    throw std::runtime_error("--classify gpu needs a positive --gpu-batch");
  if (has_chunk_files(opt.out))
    throw std::runtime_error(opt.out + " holds chunk files of a "
                             "--scheduler chunks run; use a fresh --out "
                             "directory");
  fs::create_directories(opt.out + "/jobs");

  MasterConfig cfg;
  cfg.state_path = state_path(opt.out);
  cfg.slice_seconds = opt.slice_seconds;
  cfg.checkpoint_seconds = opt.checkpoint_seconds;
  cfg.stop = &g_stop;
  cfg.progress = true;
  cfg.tty = isatty(2);

  MasterOutcome done;
  if (remote) {
    done = run_master(w, *remote, cfg);
  } else {
    Pool pool(w, nthreads, opt.budget);
    LocalRunner local(pool, &g_stop);
    done = run_master(w, local, cfg);
  }

  s.scheduler_jobs = true;
  s.jobs_done = done.fold.jobs;
  s.stopped = done.stopped;
  s.walk_complete = !done.stopped;
  s.seconds = done.seconds;
  if (done.stopped) return;
  for (size_t d = 0; d < done.fold.nodes_per_depth.size(); ++d)
    if (done.fold.nodes_per_depth[d])
      s.add_depth(static_cast<int>(d), done.fold.nodes_per_depth[d]);
  s.total_nodes = done.fold.total_nodes();
  s.max_depth_reached =
      s.nodes_per_depth.empty()
          ? 0
          : static_cast<int>(s.nodes_per_depth.size()) - 1;
  s.classified = done.fold.classified;
  s.errors = done.fold.errs;
  if (!opt.count_only)
    done.fold.types.write_csv(opt.out + "/types.csv", opt.delta);
  write_run_json(opt.out + "/run.json", s);
}
}

RunStats run(const Options& opt, SliceRunner* remote) {
  const auto t0 = std::chrono::steady_clock::now();

  Walk w = prepare(opt);

  if (!opt.scheduler_jobs && opt.classify != Classify::Cpu)
    throw std::runtime_error("--classify gpu needs --scheduler jobs");
  if (!opt.scheduler_jobs && fs::exists(state_path(opt.out)))
    throw std::runtime_error(opt.out + " holds the checkpoint of a "
                             "--scheduler jobs run; use a fresh --out "
                             "directory");
  if (opt.scheduler_jobs) fs::create_directories(opt.out);
  else fs::create_directories(opt.out + "/chunks");
  if (opt.phase == Phase::Classify) clean_tmp_files(opt.out + "/chunks");
  if (!opt.scheduler_jobs) check_existing_chunks(w, opt.out);
  write_splits_csv(opt.out + "/splits.csv", w.table);

  int nthreads = opt.threads;
  if (nthreads <= 0)
    nthreads =
        static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#else
  nthreads = 1;
#endif

  RunStats s;
  s.cli = opt.cli;
  s.delta = opt.delta;
  s.family = family_name(opt.family);
  s.drop_same_side = opt.drop_same_side;
  s.drop_trivial = opt.drop_trivial;
  s.drop_collinear = opt.drop_collinear;
  s.max_depth = opt.max_depth;
  s.threads = nthreads;
  s.count_only = opt.count_only;
  s.symmetry = opt.symmetry;
  s.sym_order = w.sym ? w.sym->order() : 1;
  s.nsplits = w.table.size();
  s.compat_pairs = w.npairs;
  s.chunks_total =
      static_cast<uint64_t>(w.chunk_count) + 1;

  g_stop.store(false);
  const StopSignals signals(opt.scheduler_jobs);

  if (opt.scheduler_jobs) {
    jobs_run(w, nthreads, remote, s);
    return s;
  }

  if (opt.phase != Phase::Classify) walk_phase(w, nthreads, s);

  const bool complete = missing_chunks(w, false).empty();
  s.walk_complete = complete;

  const bool want_classify =
      opt.phase != Phase::Walk && !opt.count_only && !s.stopped;
  if (want_classify && !complete && opt.phase == Phase::All &&
      opt.shard_count > 1) {
  } else if (want_classify) {
    classify_phase(w, s);
  }

  if (complete && !s.stopped) {
    aggregate_walk_stats(w, s);
    const bool classified = fs::exists(opt.out + "/types.csv");
    if (opt.count_only || opt.phase == Phase::Walk || classified) {
      s.seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count();
      write_run_json(opt.out + "/run.json", s);
    }
  }
  s.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  return s;
}
}
