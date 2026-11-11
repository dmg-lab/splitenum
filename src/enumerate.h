#pragma once

#include <string>

#include "compat.h"
#include "splits.h"
#include "store.h"

namespace SplitEnum {
enum class Phase { All, Walk, Classify };

struct SliceRunner;

enum class Classify { Cpu, GpuHost, Gpu, Feed };

struct Options {
  int delta = 8;
  Family family = Family::All;
  bool drop_same_side = true;
  bool drop_trivial = true;
  bool drop_collinear = false;
  bool symmetry = true;
  int max_depth = 0;
  int threads = 0;
  bool count_only = false;
  Phase phase = Phase::All;
  int chunk_pairs = 128;
  int shard_index = 0;
  int shard_count = 1;
  bool scheduler_jobs = false;
  uint64_t budget = 100000;
  double slice_seconds = 20;
  double checkpoint_seconds = 300;
  Classify classify = Classify::Cpu;
  bool refine_gpu = true;
  int gpu_batch = 65536;
  uint64_t gpu_check = 1024;
  int listen_port = -1;
  std::string connect;
  double connect_patience = 1800;
  std::string out = "out";
  std::string cli;
};

RunStats run(const Options& opt, SliceRunner* remote = nullptr);
}
