#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <string>
#include <vector>

#include "enumerate.h"
#include "jobs.h"
#include "master.h"
#include "net.h"
#include "walk.h"

namespace SplitEnum {
class TcpRunner : public SliceRunner {
 public:
  static constexpr int kSlots = 512;

  TcpRunner(const std::string& dir, int port);

  void start(const Walk& w) override;
  int slots() const override { return kSlots; }
  bool available(int slot) const override;
  void submit(int slot, std::vector<Job> jobs, double seconds) override;
  int wait(JobResult& out) override;
  void idle_wait(const std::atomic<bool>* stop) override;

  void finish();
  int connected() const;

 private:
  using Clock = std::chrono::steady_clock;

  struct Slot {
    Socket sock;
    bool busy = false;
    std::vector<Job> inflight;
    Clock::time_point deadline{};
  };

  void accept_one();
  void drop(int slot);
  bool poll_once(int timeout_ms, int& slot, JobResult& out);

  std::string dir_;
  Socket listener_;
  int port_ = 0;
  uint64_t hash_ = 0;
  std::vector<Slot> slots_;
  std::deque<int> lost_;
};

void run_worker(const Options& opt);
}
