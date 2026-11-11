#pragma once

#include <atomic>
#include <string>
#include <vector>

#include "jobs.h"
#include "pool.h"
#include "walk.h"

namespace SplitEnum {
std::atomic<bool>& stop_flag();

class StopSignals {
 public:
  explicit StopSignals(bool with_terminate);
  ~StopSignals();
  StopSignals(const StopSignals&) = delete;
  StopSignals& operator=(const StopSignals&) = delete;

 private:
  using Handler = void (*)(int);
  bool with_terminate_;
  Handler prev_int_;
  Handler prev_term_ = nullptr;
  Handler prev_usr1_ = nullptr;
};

struct SliceRunner {
  virtual ~SliceRunner() = default;
  virtual void start(const Walk&) {}
  virtual int slots() const = 0;
  virtual void submit(int slot, std::vector<Job> jobs, double seconds) = 0;
  virtual int wait(JobResult& out) = 0;
  virtual bool available(int) const { return true; }
  virtual void idle_wait(const std::atomic<bool>*) {}
};

class LocalRunner : public SliceRunner {
 public:
  LocalRunner(Pool& pool, const std::atomic<bool>* stop)
      : pool_(pool), stop_(stop) {}

  int slots() const override { return 1; }
  void submit(int slot, std::vector<Job> jobs, double seconds) override;
  int wait(JobResult& out) override;

 private:
  Pool& pool_;
  const std::atomic<bool>* stop_;
  std::vector<Job> jobs_;
  double seconds_ = 0;
};

struct MasterConfig {
  std::string state_path;
  double slice_seconds = 20;
  double checkpoint_seconds = 300;
  const std::atomic<bool>* stop = nullptr;
  bool progress = false;
  bool tty = false;
};

struct MasterOutcome {
  JobFold fold;
  double seconds = 0;
  bool stopped = false;
  bool resumed = false;
};

MasterOutcome run_master(const Walk& w, SliceRunner& runner,
                         const MasterConfig& cfg);
}
