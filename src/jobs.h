#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "store.h"
#include "walk.h"

namespace SplitEnum {
struct JobResult {
  std::vector<uint64_t> nodes_per_depth;
  std::vector<TypeInfo> types;
  uint64_t classified = 0;
  uint64_t errs = 0;
  uint64_t classify_calls = 0;
  uint64_t jobs = 0;
  bool stopped = false;
  std::vector<Job> rest;

  bool finished() const { return rest.empty(); }
  uint64_t nodes() const;
};

Job root_job(const Walk& w);

JobResult execute_job(Ctx& c, const Walk& w, const Job& job, uint64_t budget);

std::vector<Job> split_job(const Walk& w, const Job& job, int parts);

struct JobFold {
  std::vector<uint64_t> nodes_per_depth;
  TypeTable types;
  uint64_t classified = 0;
  uint64_t errs = 0;
  uint64_t classify_calls = 0;
  uint64_t jobs = 0;

  void fold(const JobResult& r);
  uint64_t total_nodes() const;
  JobResult snapshot() const;
};

struct JobBatch {
  std::vector<Job> jobs;
  double seconds = 0;
};

struct JobState {
  JobResult done;
  double seconds = 0;
};

uint64_t job_config_hash(const Walk& w);

std::vector<uint8_t> encode_job(const Job& job, uint64_t config_hash);
Job decode_job(const std::vector<uint8_t>& bytes, uint64_t config_hash);

std::vector<uint8_t> encode_result(const JobResult& r, uint64_t config_hash);
JobResult decode_result(const std::vector<uint8_t>& bytes,
                        uint64_t config_hash);

std::vector<uint8_t> encode_jobs(const JobBatch& batch, uint64_t config_hash);
JobBatch decode_jobs(const std::vector<uint8_t>& bytes, uint64_t config_hash);

std::vector<uint8_t> encode_state(const JobState& state, uint64_t config_hash);
JobState decode_state(const std::vector<uint8_t>& bytes, uint64_t config_hash);
}
