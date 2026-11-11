#include <mpi.h>

#include <chrono>
#include <cstdio>
#include <exception>
#include <thread>
#include <vector>

#include "cli.h"
#include "gpuclassify.h"
#include "jobs.h"
#include "master.h"
#include "pool.h"
#include "walk.h"

namespace {
using namespace SplitEnum;

constexpr int kTagJobs = 1;
constexpr int kTagResult = 2;
constexpr int kTagStop = 3;

void send_bytes(const std::vector<uint8_t>& bytes, int dest, int tag) {
  MPI_Send(bytes.data(), static_cast<int>(bytes.size()), MPI_BYTE, dest, tag,
           MPI_COMM_WORLD);
}

std::vector<uint8_t> recv_bytes(int source, int tag, MPI_Status& status) {
  int flag = 0;
  while (true) {
    MPI_Iprobe(source, tag, MPI_COMM_WORLD, &flag, &status);
    if (flag) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  int size = 0;
  MPI_Get_count(&status, MPI_BYTE, &size);
  std::vector<uint8_t> bytes(size);
  MPI_Recv(bytes.data(), size, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  return bytes;
}

class MpiRunner : public SliceRunner {
 public:
  explicit MpiRunner(int workers) : workers_(workers) {}

  void start(const Walk& w) override { hash_ = job_config_hash(w); }
  int slots() const override { return workers_; }

  void submit(int slot, std::vector<Job> jobs, double seconds) override {
    JobBatch batch;
    batch.jobs = std::move(jobs);
    batch.seconds = seconds;
    send_bytes(encode_jobs(batch, hash_), slot + 1, kTagJobs);
  }

  int wait(JobResult& out) override {
    MPI_Status status;
    const std::vector<uint8_t> bytes =
        recv_bytes(MPI_ANY_SOURCE, kTagResult, status);
    out = decode_result(bytes, hash_);
    return status.MPI_SOURCE - 1;
  }

  void shutdown() {
    for (int r = 1; r <= workers_; ++r)
      MPI_Send(nullptr, 0, MPI_BYTE, r, kTagStop, MPI_COMM_WORLD);
  }

 private:
  int workers_;
  uint64_t hash_ = 0;
};

void work(const Options& opt) {
  const StopSignals signals(true);
  const Walk w = prepare(opt);
  const uint64_t hash = job_config_hash(w);
  int nthreads = opt.threads;
  if (nthreads <= 0)
    nthreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  Pool pool(w, nthreads, opt.budget);
  while (true) {
    MPI_Status status;
    const std::vector<uint8_t> bytes = recv_bytes(0, MPI_ANY_TAG, status);
    if (status.MPI_TAG == kTagStop) {
      print_batch_stats(stderr);
      return;
    }
    JobBatch batch = decode_jobs(bytes, hash);
    const JobResult r =
        pool.run(std::move(batch.jobs), batch.seconds, &stop_flag());
    send_bytes(encode_result(r, hash), 0, kTagResult);
  }
}
}

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Options opt;
  const int rc = parse_cli(argc, argv, opt);
  if (rc >= 0) {
    MPI_Finalize();
    return rc;
  }
  opt.scheduler_jobs = true;

  try {
    if (size < 2)
      throw std::runtime_error("splitenum-mpi needs at least 2 ranks: one "
                               "master and one worker");
    if (rank == 0) {
      MpiRunner runner(size - 1);
      const RunStats s = run(opt, &runner);
      runner.shutdown();
      report(s, opt);
    } else {
      work(opt);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "splitenum-mpi rank %d: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
  return 0;
}
