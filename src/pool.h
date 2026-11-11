#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "jobs.h"
#include "walk.h"

namespace SplitEnum {
class Pool {
 public:
  Pool(const Walk& w, int threads, uint64_t budget);

  int threads() const { return static_cast<int>(ctxs_.size()); }

  JobResult run(std::vector<Job> seed, double seconds,
                const std::atomic<bool>* stop);

 private:
  const Walk& w_;
  uint64_t budget_;
  std::vector<Ctx> ctxs_;
};
}
