#include "cli.h"

#include "gpuclassify.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>

namespace SplitEnum {
namespace {
const char* kUsage =
    "splitenum -- enumerate compatible collections of Harnack splits\n"
    "\n"
    "Walks the tree whose root is the empty collection and whose children add\n"
    "one split of strictly larger index that stays compatible with everything\n"
    "already chosen.  Records the isotopy type of every node.\n"
    "\n"
    "Every visited node is classified exactly, one orbit representative\n"
    "each.  With --scheduler chunks the walk is cut into chunks of depth-2\n"
    "pairs, each written as one per-type table under DIR/chunks, and the\n"
    "final phase folds those into types.csv; with --scheduler jobs a master\n"
    "hands out budgeted jobs and checkpoints DIR/jobs/state.bin.  Either way\n"
    "the run may be stopped at any time, Ctrl-C or a hard kill alike, and\n"
    "rerunning the same command continues where it left off.\n"
    "\n"
    "usage: splitenum [options]\n"
    "\n"
    "  --delta N            degree of the simplex (default 8, max 9)\n"
    "  --family F           all | with-zero | without-zero  (default all)\n"
    "                       with-zero keeps splits whose parity pair contains\n"
    "                       (0,0), without-zero the others\n"
    "  --same-side drop|keep\n"
    "                       simple splits with both endpoints on one boundary\n"
    "                       edge (default drop)\n"
    "  --trivial drop|keep  splits whose twist is boundary-point flips XOR an\n"
    "                       affine mask; they never change the real scheme\n"
    "                       (default drop; split ids and node counts are not\n"
    "                       comparable across this flag)\n"
    "  --collinear keep|drop\n"
    "                       collinear double splits, straight cuts subdivided\n"
    "                       at an interior midpoint (default keep; they are\n"
    "                       required for completeness; split ids and node\n"
    "                       counts are not comparable across this flag)\n"
    "  --symmetry on|off    prune the S_3 symmetry of the simplex: walk only\n"
    "                       orbit-canonical collections, with exact orbit-\n"
    "                       weighted node counts (default on; types.csv is\n"
    "                       unchanged by the flag)\n"
    "  --max-depth N        stop at collections of size N (default unlimited)\n"
    "  --threads T          default: hardware concurrency\n"
    "  --phase P            walk | classify | all  (default all)\n"
    "  --chunk-pairs N      depth-2 pairs per walk chunk (default 128)\n"
    "  --shard I/N          walk only the chunks with id %% N == I; copy the\n"
    "                       shards' DIR/chunks together, then --phase classify\n"
    "  --scheduler chunks|jobs\n"
    "                       chunks (default): fixed depth-2 pair chunks, one\n"
    "                       file each; jobs: a master hands budgeted jobs to\n"
    "                       the worker threads, splits them to keep every\n"
    "                       thread busy and checkpoints DIR/jobs/state.bin;\n"
    "                       the two refuse to share a directory.  splitenum-mpi runs\n"
    "                       the jobs scheduler across MPI ranks\n"
    "  --budget N           jobs: candidate visits per job before it yields\n"
    "                       its remainder (default 100000)\n"
    "  --slice-seconds S    jobs: longest stretch a worker pool runs before\n"
    "                       reporting back (default 20)\n"
    "  --checkpoint-seconds S\n"
    "                       jobs: interval between checkpoints (default 300)\n"
    "  --classify cpu|gpu-host|gpu\n"
    "                       jobs only: cpu (default) classifies\n"
    "                       node by node with Isotopy::Graph; gpu batches\n"
    "                       the refined triangulations through the CUDA\n"
    "                       kernel of libisotopy, one device per thread\n"
    "                       round robin, and needs splitenum-gpu; gpu-host\n"
    "                       runs the same kernel code on the CPU\n"
    "  --refine cpu|gpu     with --classify gpu|gpu-host: where the\n"
    "                       triangulation sweep runs; gpu ships split ids\n"
    "                       instead of edges and keeps the cards busy\n"
    "                       (default gpu; ignored by --classify cpu)\n"
    "  --gpu-batch N        nodes per kernel launch (default 65536)\n"
    "  --gpu-check N        every N-th batch is also classified on the CPU\n"
    "                       and compared; 0 disables (default 1024)\n"
    "  --listen PORT        jobs: be the master of a network run, accept\n"
    "                       workers on PORT (0 picks a free one) and write\n"
    "                       DIR/jobs/master.addr; walks nothing itself\n"
    "  --connect DIR        be a worker of the master whose --out is DIR:\n"
    "                       read DIR/jobs/master.addr, run slices with\n"
    "                       --threads threads, reconnect after a lost\n"
    "                       master; implies --scheduler jobs\n"
    "  --connect-patience S seconds a worker waits for an absent master\n"
    "                       before giving up (default 1800)\n"
    "  --count-only         count the tree, touch no isotopy machinery\n"
    "  --out DIR            output directory (default out)\n"
    "  -h, --help           this text\n";

struct Toggle {
  const char* flag;
  const char* yes;
  const char* no;
  bool* target;
};

bool need_value(int argc, int i, const char* flag) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "splitenum: %s needs a value\n", flag);
    return false;
  }
  return true;
}
}

int parse_cli(int argc, char** argv, Options& opt) {
  for (int i = 0; i < argc; ++i) {
    if (i) opt.cli += ' ';
    opt.cli += argv[i];
  }

  const Toggle toggles[] = {
      {"--same-side", "drop", "keep", &opt.drop_same_side},
      {"--trivial", "drop", "keep", &opt.drop_trivial},
      {"--collinear", "drop", "keep", &opt.drop_collinear},
      {"--symmetry", "on", "off", &opt.symmetry},
  };

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      std::fputs(kUsage, stdout);
      return 0;
    }
    if (a == "--count-only") { opt.count_only = true; continue; }
    if (!need_value(argc, i, argv[i])) return 2;
    const std::string v = argv[++i];
    const Toggle* t = std::find_if(std::begin(toggles), std::end(toggles),
                                   [&](const Toggle& x) { return a == x.flag; });
    if (t != std::end(toggles)) {
      if (v != t->yes && v != t->no) {
        std::fprintf(stderr, "splitenum: %s %s|%s\n", t->flag, t->yes, t->no);
        return 2;
      }
      *t->target = v == t->yes;
      continue;
    }
    if (a == "--delta") opt.delta = std::atoi(v.c_str());
    else if (a == "--family") {
      if (!parse_family(v, opt.family)) {
        std::fprintf(stderr, "splitenum: unknown family '%s'\n", v.c_str());
        return 2;
      }
    } else if (a == "--max-depth") opt.max_depth = std::atoi(v.c_str());
    else if (a == "--threads") opt.threads = std::atoi(v.c_str());
    else if (a == "--phase") {
      if (v == "walk") opt.phase = Phase::Walk;
      else if (v == "classify") opt.phase = Phase::Classify;
      else if (v == "all") opt.phase = Phase::All;
      else { std::fprintf(stderr, "splitenum: --phase walk|classify|all\n"); return 2; }
    } else if (a == "--chunk-pairs") opt.chunk_pairs = std::atoi(v.c_str());
    else if (a == "--shard") {
      const size_t slash = v.find('/');
      if (slash == std::string::npos) {
        std::fprintf(stderr, "splitenum: --shard I/N\n");
        return 2;
      }
      opt.shard_index = std::atoi(v.substr(0, slash).c_str());
      opt.shard_count = std::atoi(v.substr(slash + 1).c_str());
    } else if (a == "--scheduler") {
      if (v != "chunks" && v != "jobs") {
        std::fprintf(stderr, "splitenum: --scheduler chunks|jobs\n");
        return 2;
      }
      opt.scheduler_jobs = v == "jobs";
    } else if (a == "--budget")
      opt.budget = std::strtoull(v.c_str(), nullptr, 10);
    else if (a == "--slice-seconds") opt.slice_seconds = std::atof(v.c_str());
    else if (a == "--checkpoint-seconds")
      opt.checkpoint_seconds = std::atof(v.c_str());
    else if (a == "--classify") {
      if (v == "cpu") opt.classify = Classify::Cpu;
      else if (v == "gpu-host") opt.classify = Classify::GpuHost;
      else if (v == "gpu") opt.classify = Classify::Gpu;
      else {
        std::fprintf(stderr, "splitenum: --classify cpu|gpu-host|gpu\n");
        return 2;
      }
    } else if (a == "--refine") {
      if (v == "cpu") opt.refine_gpu = false;
      else if (v == "gpu") opt.refine_gpu = true;
      else {
        std::fprintf(stderr, "splitenum: --refine cpu|gpu\n");
        return 2;
      }
    } else if (a == "--gpu-batch") opt.gpu_batch = std::atoi(v.c_str());
    else if (a == "--gpu-check")
      opt.gpu_check = std::strtoull(v.c_str(), nullptr, 10);
    else if (a == "--listen") {
      opt.listen_port = std::atoi(v.c_str());
      opt.scheduler_jobs = true;
    } else if (a == "--connect") {
      opt.connect = v;
      opt.scheduler_jobs = true;
    } else if (a == "--connect-patience")
      opt.connect_patience = std::atof(v.c_str());
    else if (a == "--out") opt.out = v;
    else {
      std::fprintf(stderr, "splitenum: unknown option '%s'\n", a.c_str());
      return 2;
    }
  }

  return -1;
}

void report(const RunStats& s, const Options& opt) {
  std::printf("delta %d, family %s, same-side %s, trivial %s, collinear %s\n",
              s.delta, s.family.c_str(), s.drop_same_side ? "dropped" : "kept",
              s.drop_trivial ? "dropped" : "kept",
              s.drop_collinear ? "dropped" : "kept");
  std::printf("splits            %d\n", s.nsplits);
  std::printf("compatible pairs  %llu\n",
              static_cast<unsigned long long>(s.compat_pairs));
  if (s.symmetry)
    std::printf("symmetry          on, group order %d\n", s.sym_order);
  else
    std::printf("symmetry          off\n");
  if (s.scheduler_jobs)
    std::printf("jobs              %llu executed in %.2fs\n",
                static_cast<unsigned long long>(s.jobs_done), s.seconds);
  else
    std::printf("chunks            %llu walked now, %llu assigned, in %.2fs "
                "on %d threads\n",
                static_cast<unsigned long long>(s.chunks_done),
                static_cast<unsigned long long>(s.chunks_total), s.seconds,
                s.threads);
  print_batch_stats(stdout);
  if (s.stopped) {
    std::printf("stopped early -- rerun the same command to continue\n");
    return;
  }
  if (!s.walk_complete) {
    std::printf("walk incomplete (other shards pending); classify runs once "
                "every chunk is present\n");
    return;
  }
  std::printf("nodes             %llu\n",
              static_cast<unsigned long long>(s.total_nodes));
  std::printf("max depth reached %d\n", s.max_depth_reached);
  for (size_t d = 0; d < s.nodes_per_depth.size(); ++d)
    std::printf("  depth %2zu  %llu\n", d,
                static_cast<unsigned long long>(s.nodes_per_depth[d]));
  if (!s.count_only && s.classified) {
    std::printf("classified        %llu representatives (%.1fx orbit collapse)\n",
                static_cast<unsigned long long>(s.classified),
                static_cast<double>(s.total_nodes) / s.classified);
    if (s.errors)
      std::printf("errors            %llu (recorded as type #ERR)\n",
                  static_cast<unsigned long long>(s.errors));
    std::printf("results in        %s/types.csv, %s/run.json\n",
                opt.out.c_str(), opt.out.c_str());
  }
}
}
