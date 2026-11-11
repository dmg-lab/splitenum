#include "master.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include <unistd.h>

namespace fs = std::filesystem;

namespace SplitEnum {
namespace {
void on_stop_signal(int) {
  stop_flag().store(true);
  const char msg[] =
      "\nsplitenum: finishing in-flight work, then stopping; rerun the same "
      "command to continue (a hard kill is safe too)\n";
  [[maybe_unused]] const auto n = write(2, msg, sizeof msg - 1);
}

std::vector<uint8_t> read_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),
                              std::istreambuf_iterator<char>());
}

void write_bytes_atomic(const std::string& path,
                        const std::vector<uint8_t>& bytes) {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    f.flush();
    if (!f) throw std::runtime_error("cannot write " + tmp);
  }
  fs::rename(tmp, path);
}
}

std::atomic<bool>& stop_flag() {
  static std::atomic<bool> flag{false};
  return flag;
}

StopSignals::StopSignals(bool with_terminate)
    : with_terminate_(with_terminate),
      prev_int_(std::signal(SIGINT, on_stop_signal)) {
  if (with_terminate_) {
    prev_term_ = std::signal(SIGTERM, on_stop_signal);
    prev_usr1_ = std::signal(SIGUSR1, on_stop_signal);
  }
}

StopSignals::~StopSignals() {
  std::signal(SIGINT, prev_int_);
  if (with_terminate_) {
    std::signal(SIGTERM, prev_term_);
    std::signal(SIGUSR1, prev_usr1_);
  }
}

void LocalRunner::submit(int, std::vector<Job> jobs, double seconds) {
  jobs_ = std::move(jobs);
  seconds_ = seconds;
}

int LocalRunner::wait(JobResult& out) {
  out = pool_.run(std::move(jobs_), seconds_, stop_);
  jobs_.clear();
  return 0;
}

MasterOutcome run_master(const Walk& w, SliceRunner& runner,
                         const MasterConfig& cfg) {
  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  const auto since = [](Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
  };
  const uint64_t hash = job_config_hash(w);

  MasterOutcome out;
  std::vector<Job> pending;
  double carried = 0;
  if (!cfg.state_path.empty() && fs::exists(cfg.state_path)) {
    JobState st = decode_state(read_bytes(cfg.state_path), hash);
    pending = std::move(st.done.rest);
    st.done.rest.clear();
    out.fold.fold(st.done);
    carried = st.seconds;
    out.resumed = true;
  } else {
    pending.push_back(root_job(w));
  }

  const auto checkpoint = [&] {
    if (cfg.state_path.empty()) return;
    JobState st;
    st.done = out.fold.snapshot();
    st.done.rest = pending;
    st.seconds = carried + since(t0);
    write_bytes_atomic(cfg.state_path, encode_state(st, hash));
  };

  runner.start(w);
  const int slots = runner.slots();
  std::vector<char> idle(slots, 1);
  int outstanding = 0;
  bool stopping = false;
  const double max_slice = std::max(cfg.slice_seconds, 1e-4);
  double slice = std::min(0.5, max_slice);
  auto last_checkpoint = Clock::now();
  auto last_progress = Clock::now();
  const uint64_t nodes0 = out.fold.total_nodes();

  const auto assign = [&] {
    std::vector<int> free;
    for (int s = 0; s < slots; ++s)
      if (idle[s] && runner.available(s)) free.push_back(s);
    if (free.empty() || pending.empty()) return;
    std::stable_sort(pending.begin(), pending.end(),
                     [](const Job& a, const Job& b) {
                       return a.prefix.size() < b.prefix.size();
                     });
    while (pending.size() < free.size()) {
      std::vector<Job> parts = split_job(
          w, pending.front(), static_cast<int>(free.size() - pending.size() + 1));
      if (parts.size() < 2) break;
      pending.erase(pending.begin());
      pending.insert(pending.begin(), std::make_move_iterator(parts.begin()),
                     std::make_move_iterator(parts.end()));
    }
    const size_t k = std::min(free.size(), pending.size());
    const bool starving = pending.size() < free.size() + static_cast<size_t>(outstanding);
    slice = starving ? std::max(std::min(0.5, max_slice), slice / 2)
                     : std::min(max_slice, slice * 2);
    std::vector<std::vector<Job>> hands(k);
    for (size_t i = 0; i < pending.size(); ++i)
      hands[i % k].push_back(std::move(pending[i]));
    pending.clear();
    for (size_t i = 0; i < k; ++i) {
      idle[free[i]] = 0;
      ++outstanding;
      runner.submit(free[i], std::move(hands[i]), slice);
    }
  };

  bool fresh = true;
  while (true) {
    if (cfg.stop && cfg.stop->load(std::memory_order_relaxed)) stopping = true;
    const bool due =
        !fresh && since(last_checkpoint) >= cfg.checkpoint_seconds;
    fresh = false;
    if (!stopping && !due) assign();
    if (outstanding == 0) {
      if (pending.empty()) break;
      if (stopping) {
        out.stopped = true;
        break;
      }
      if (!due) {
        runner.idle_wait(cfg.stop);
        continue;
      }
      checkpoint();
      last_checkpoint = Clock::now();
      fresh = true;
      continue;
    }
    JobResult r;
    const int slot = runner.wait(r);
    idle[slot] = 1;
    --outstanding;
    if (r.stopped) stopping = true;
    for (Job& j : r.rest) pending.push_back(std::move(j));
    r.rest.clear();
    out.fold.fold(r);
    if (cfg.progress && since(last_progress) >= (cfg.tty ? 1.0 : 60.0)) {
      last_progress = Clock::now();
      const uint64_t nodes = out.fold.total_nodes();
      std::fprintf(stderr, "%sjobs: %llu nodes, %llu classified, %zu pending, "
                           "%.3g nodes/s%s",
                   cfg.tty ? "\r" : "", static_cast<unsigned long long>(nodes),
                   static_cast<unsigned long long>(out.fold.classify_calls),
                   pending.size(),
                   static_cast<double>(nodes - nodes0) / since(t0),
                   cfg.tty ? "   " : "\n");
    }
  }
  if (cfg.progress && cfg.tty) std::fprintf(stderr, "\n");
  checkpoint();
  out.seconds = carried + since(t0);
  return out;
}
}
