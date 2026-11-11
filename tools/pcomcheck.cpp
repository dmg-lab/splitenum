#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "isotopy_graph.h"

const char* kUsage =
    "usage: pcomcheck DELTA < FILE\n"
    "\n"
    "Prints the real scheme libisotopy computes for a patchwork, as\n"
    "\"viro p n is_mcurve\".  FILE holds the sign vector on line 1, one 0 or 1\n"
    "per lattice point in flat order, then one triangle per line as three\n"
    "flat vertex indices, the format of the .pcom files of\n"
    "github.com/dmg-lab/CombinatorialPatchworking.\n";

int main(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
    std::fputs(kUsage, stderr);
    return 2;
  }
  const int delta = std::atoi(argv[1]);

  std::string bits;
  if (!(std::cin >> bits)) {
    std::fputs("pcomcheck: no sign line\n", stderr);
    return 2;
  }
  std::vector<bool> signs;
  signs.reserve(bits.size());
  for (const char c : bits) signs.push_back(c == '1');

  std::vector<Isotopy::Triangle> tris;
  int a, b, c;
  while (std::cin >> a >> b >> c) tris.push_back({a, b, c});

  std::fprintf(stderr, "pcomcheck: delta %d, %zu signs, %zu triangles\n", delta,
               signs.size(), tris.size());
  try {
    Isotopy::Graph g(delta, signs, tris);
    const std::string v = g.viro_notation();
    std::printf("viro      %s\n", v.c_str());
    std::printf("p         %d\n", g.p_regions);
    std::printf("n         %d\n", g.n_regions);
    std::printf("is_mcurve %d\n", g.is_mcurve() ? 1 : 0);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "pcomcheck: %s\n", e.what());
    return 1;
  }
  return 0;
}
