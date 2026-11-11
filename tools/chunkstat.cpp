#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "chunkio.h"

namespace fs = std::filesystem;

const char* kUsage =
    "usage: chunkstat DIR/chunks [...]\n"
    "\n"
    "Prints one TSV row per walk chunk file of a --scheduler chunks run:\n"
    "index pair_lo pair_hi seconds records nodes depth.  The seconds per\n"
    "chunk are what --chunk-pairs should be tuned on; --chunk-pairs is\n"
    "fixed for the life of an output directory.\n";

int main(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
    std::fputs(kUsage, stderr);
    return 2;
  }

  std::vector<std::string> paths;
  for (int i = 1; i < argc; ++i) {
    const fs::path p = argv[i];
    if (fs::is_directory(p)) {
      for (const auto& e : fs::directory_iterator(p)) {
        const std::string n = e.path().filename().string();
        if (n.rfind("walk-", 0) == 0 && e.path().extension() == ".bin")
          paths.push_back(e.path().string());
      }
    } else {
      paths.push_back(p.string());
    }
  }
  std::sort(paths.begin(), paths.end());

  std::printf("index\tpair_lo\tpair_hi\tseconds\trecords\tnodes\tdepth\n");
  for (const std::string& p : paths) {
    try {
      const SplitEnum::ChunkHeader h = SplitEnum::read_chunk_header(p);
      unsigned long long nodes = 0;
      for (const uint64_t n : h.nodes_per_depth) nodes += n;
      std::printf("%d\t%llu\t%llu\t%.3f\t%llu\t%llu\t%zu\n", h.chunk_index,
                  static_cast<unsigned long long>(h.pair_lo),
                  static_cast<unsigned long long>(h.pair_hi), h.seconds,
                  static_cast<unsigned long long>(h.nrecords), nodes,
                  h.nodes_per_depth.empty() ? 0 : h.nodes_per_depth.size() - 1);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "chunkstat: %s: %s\n", p.c_str(), e.what());
    }
  }
  return 0;
}
