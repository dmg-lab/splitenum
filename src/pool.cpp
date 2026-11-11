#include "pool.h"

#include "gpuclassify.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace SplitEnum {
namespace {
class DepthQueue {
 public:
  size_t size() const { return size_; }

  void push(Job j) {
    const size_t d = j.prefix.size();
    if (d >= buckets_.size()) buckets_.resize(d + 1);
    buckets_[d].push_back(std::move(j));
    ++size_;
  }

  Job pop_shallowest() {
    size_t d = 0;
    while (buckets_[d].empty()) ++d;
    return take(d);
  }

  Job pop_deepest() {
    size_t d = buckets_.size() - 1;
    while (buckets_[d].empty()) --d;
    return take(d);
  }

  std::vector<Job> drain() {
    std::vector<Job> out;
    out.reserve(size_);
    for (auto& b : buckets_) {
      for (Job& j : b) out.push_back(std::move(j));
      b.clear();
    }
    size_ = 0;
    return out;
  }

 private:
  Job take(size_t d) {
    Job j = std::move(buckets_[d].back());
    buckets_[d].pop_back();
    --size_;
    return j;
  }

  std::vector<std::vector<Job>> buckets_;
  size_t size_ = 0;
};
}

Pool::Pool(const Walk& w, int threads, uint64_t budget)
    : w_(w), budget_(budget), ctxs_(std::max(threads, 1)) {
  if (budget == 0) throw std::invalid_argument("job budget must be positive");
  for (Ctx& c : ctxs_) c.init(w);
  const Options& opt = *w.opt;
  if (opt.classify == Classify::Cpu || opt.count_only) return;
  if (!w.lean)
    throw std::logic_error("lean refine missing for a classifying pool");
  for (size_t t = 0; t < ctxs_.size(); ++t) {
    ctxs_[t].batch = std::make_shared<BatchClassifier>(
        w.table, *w.lean, opt.classify, static_cast<int>(t), opt.gpu_batch,
        opt.gpu_check, opt.refine_gpu);
    ctxs_[t].batch->pin();
    ctxs_[t].batch_defer = true;
  }
}

JobResult Pool::run(std::vector<Job> seed, double seconds,
                    const std::atomic<bool>* stop) {
  using Clock = std::chrono::steady_clock;
  const int n = threads();
  const uint64_t ramp_budget = std::max<uint64_t>(budget_ / 100, 1);

  std::mutex m;
  std::condition_variable cv;
  DepthQueue queue;
  for (Job& j : seed) queue.push(std::move(j));
  int busy = 0;
  bool ending = false;
  std::atomic<bool> abort{false};
  std::exception_ptr failure;
  std::vector<JobFold> folds(n);

  const auto work = [&](int t) {
    Ctx& c = ctxs_[t];
    c.abort = &abort;
    std::unique_lock<std::mutex> lk(m);
    while (true) {
      cv.wait(lk, [&] { return ending || queue.size() > 0 || busy == 0; });
      if (ending || queue.size() == 0) break;
      const bool low = queue.size() < static_cast<size_t>(2 * n);
      Job job = low ? queue.pop_shallowest() : queue.pop_deepest();
      const int idle = queue.size() == 0 ? n - 1 - busy : 0;
      ++busy;
      lk.unlock();
      JobResult r;
      try {
        if (idle > 0) {
          std::vector<Job> parts = split_job(w_, job, idle + 1);
          job = std::move(parts.front());
          if (parts.size() > 1) {
            lk.lock();
            for (size_t i = 1; i < parts.size(); ++i)
              queue.push(std::move(parts[i]));
            lk.unlock();
            cv.notify_all();
          }
        }
        r = execute_job(c, w_, job, low ? ramp_budget : budget_);
      } catch (...) {
        if (!lk.owns_lock()) lk.lock();
        --busy;
        if (!failure) failure = std::current_exception();
        ending = true;
        break;
      }
      folds[t].fold(r);
      lk.lock();
      --busy;
      for (Job& x : r.rest) queue.push(std::move(x));
      cv.notify_all();
    }
    c.abort = nullptr;
    lk.unlock();
    if (c.batch) {
      try {
        TypeTable tail;
        JobResult r;
        c.batch->flush(tail, r.errs);
        r.types = tail.types();
        folds[t].fold(r);
      } catch (...) {
        lk.lock();
        if (!failure) failure = std::current_exception();
        lk.unlock();
      }
    }
    cv.notify_all();
  };

  std::vector<std::thread> team;
  team.reserve(n);
  for (int t = 0; t < n; ++t) team.emplace_back(work, t);

  bool stopped = false;
  {
    const auto deadline =
        Clock::now() + std::chrono::duration_cast<Clock::duration>(
                           std::chrono::duration<double>(seconds));
    std::unique_lock<std::mutex> lk(m);
    while (!ending && !(queue.size() == 0 && busy == 0)) {
      if (stop && stop->load(std::memory_order_relaxed)) {
        stopped = true;
        break;
      }
      const auto now = Clock::now();
      if (now >= deadline) break;
      cv.wait_until(lk, std::min(deadline, now + std::chrono::milliseconds(20)));
    }
    ending = true;
    abort.store(true);
  }
  cv.notify_all();
  for (std::thread& th : team) th.join();
  if (failure) std::rethrow_exception(failure);

  JobFold total;
  for (const JobFold& f : folds) total.fold(f.snapshot());
  JobResult out = total.snapshot();
  out.rest = queue.drain();
  out.stopped = stopped;
  return out;
}
}
