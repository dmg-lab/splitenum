#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "enumerate.h"
#include "jobs.h"

const char* kUsage =
    "usage: statepeek DIR/jobs/state.bin [--delta N] [--write DIR]\n"
    "\n"
    "Decodes a jobs checkpoint with the default split table of degree N,\n"
    "default 9, and prints nodes, representatives, errors and pending jobs,\n"
    "then the per-scheme totals folded so far.  --write DIR also writes\n"
    "types.csv, splits.csv and a partial run.json there, the inputs of\n"
    "bin/witnesses, so witnesses of an unfinished run can be verified.\n";

int main(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
    std::fputs(kUsage, stderr);
    return 2;
  }
  SplitEnum::Options opt;
  opt.delta = 9;
  std::string write_dir;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--delta") == 0) opt.delta = std::atoi(argv[i + 1]);
    if (std::strcmp(argv[i], "--write") == 0) write_dir = argv[i + 1];
  }
  std::ifstream f(argv[1], std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(f),
                             std::istreambuf_iterator<char>()};
  SplitEnum::Walk w = SplitEnum::prepare(opt);
  SplitEnum::JobState st =
      SplitEnum::decode_state(bytes, SplitEnum::job_config_hash(w));
  std::map<std::string, uint64_t> per_viro;
  for (const auto& t : st.done.types) per_viro[t.viro] += t.count;
  std::printf("nodes %llu\nclassified %llu\nerrors %llu\npending_jobs %zu\n"
              "seconds %.0f\nrows %zu\nschemes %zu\n",
              (unsigned long long)st.done.nodes(),
              (unsigned long long)st.done.classified,
              (unsigned long long)st.done.errs, st.done.rest.size(),
              st.seconds, st.done.types.size(), per_viro.size());
  std::printf("viro,count\n");
  for (const auto& [viro, count] : per_viro)
    std::printf("\"%s\",%llu\n", viro.c_str(), (unsigned long long)count);
  if (write_dir.empty()) return 0;
  std::filesystem::create_directories(write_dir);
  SplitEnum::TypeTable table;
  for (const auto& t : st.done.types)
    table.add(t.viro, t.p, t.n, t.mcurve, t.profile, t.count, t.first_depth,
              t.witness.data(), t.witness.size());
  table.write_csv(write_dir + "/types.csv", opt.delta);
  SplitEnum::write_splits_csv(write_dir + "/splits.csv", w.table);
  SplitEnum::RunStats stats;
  stats.cli = "statepeek, partial checkpoint";
  stats.delta = opt.delta;
  stats.family = SplitEnum::family_name(opt.family);
  stats.drop_same_side = opt.drop_same_side;
  stats.drop_trivial = opt.drop_trivial;
  stats.drop_collinear = opt.drop_collinear;
  stats.symmetry = opt.symmetry;
  stats.nsplits = w.table.size();
  stats.nodes_per_depth = st.done.nodes_per_depth;
  stats.total_nodes = st.done.nodes();
  stats.classified = st.done.classified;
  stats.errors = st.done.errs;
  stats.seconds = st.seconds;
  stats.scheduler_jobs = true;
  stats.stopped = true;
  SplitEnum::write_run_json(write_dir + "/run.json", stats);
  std::fprintf(stderr, "wrote %s/{types.csv,splits.csv,run.json}\n", write_dir.c_str());
  return 0;
}
