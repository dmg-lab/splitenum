#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "splits.h"

namespace SplitEnum {
struct TypeInfo {
  std::string viro;
  int p = 0;
  int n = 0;
  bool mcurve = false;
  Profile profile = kEmptyProfile;
  uint64_t count = 0;
  int first_depth = 0;
  std::vector<uint16_t> witness;
};

class TypeTable {
 public:
  void add(const std::string& viro, int p, int n, bool mcurve, Profile profile,
           uint64_t count, int depth, const uint16_t* stack, size_t stack_len);
  const std::vector<TypeInfo>& types() const { return types_; }
  void write_csv(const std::string& path, int delta) const;

 private:
  std::unordered_map<std::string, uint32_t> id_of_;
  std::vector<TypeInfo> types_;
};

struct RunStats {
  std::string cli;
  int delta = 0;
  std::string family;
  bool drop_same_side = true;
  bool drop_trivial = true;
  bool drop_collinear = false;
  int max_depth = 0;
  int threads = 1;
  bool count_only = false;
  bool symmetry = true;
  int sym_order = 1;
  int nsplits = 0;
  uint64_t compat_pairs = 0;
  std::vector<uint64_t> nodes_per_depth;
  uint64_t total_nodes = 0;
  int max_depth_reached = 0;
  uint64_t classified = 0;
  uint64_t errors = 0;
  double seconds = 0.0;
  uint64_t chunks_done = 0;
  uint64_t chunks_total = 0;
  bool scheduler_jobs = false;
  uint64_t jobs_done = 0;
  bool walk_complete = false;
  bool stopped = false;

  void add_depth(int d, uint64_t k);
};

void write_run_json(const std::string& path, const RunStats& s);

void write_splits_csv(const std::string& path, const SplitTable& table);
}
