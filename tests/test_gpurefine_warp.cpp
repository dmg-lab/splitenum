#define ISOTOPY_GPU_HOST_WARP
#include "catch.hpp"

#include <random>
#include <thread>
#include <vector>

#include "compat.h"
#include "gpuclassify.h"
#include "gpurefine.h"
#include "refine.h"
#include "splits.h"

TEST_CASE("the 32-lane warp sweep reproduces LeanRefine byte for byte",
          "[refine]") {
  constexpr int kTop = MAX_DELTA < 9 ? MAX_DELTA : 9;
  for (int delta : {6, kTop}) {
    INFO("delta " << delta);
    const SplitEnum::SplitTable t = SplitEnum::build_splits(
        delta, SplitEnum::Family::All, true, false, false);
    const SplitEnum::Compat c = SplitEnum::build_compat(t);
    const SplitEnum::LeanRefine lean(t);
    const SplitEnum::RefineTables rt = SplitEnum::refine_tables(lean);
    const size_t nbytes = 2 * static_cast<size_t>(lean.nedges());
    std::vector<uint8_t> want(nbytes), got(nbytes);
    std::mt19937 rng(7 * delta);
    int refined = 0;
    for (int i = 0; i < 500; ++i) {
      std::vector<uint16_t> ids;
      const size_t target = i % 4 == 3 ? 3 : 1 + rng() % 30;
      for (int tries = 0; tries < 4 * t.size() && ids.size() < target; ++tries) {
        const uint16_t v = static_cast<uint16_t>(rng() % t.size());
        bool ok = i % 4 == 3;
        if (!ok) {
          ok = true;
          for (uint16_t u : ids) ok = ok && u != v && c.get(u, v);
        }
        if (ok) ids.push_back(v);
      }
      const int n = static_cast<int>(ids.size());
      uint64_t acc[SplitEnum::LeanRefine::kWords];
      const bool cpu = lean.refine(ids.data(), n, acc);
      SplitEnum::RefineScratch rs;
      IsotopyGPU::HostWarpContext ctx;
      bool lane_ok[32];
      std::vector<std::thread> lanes;
      for (int l = 0; l < 32; ++l)
        lanes.emplace_back([&, l] {
          const IsotopyGPU::Warp<32> w{&ctx, l};
          lane_ok[l] = SplitEnum::sweep_warp(rt, ids.data(), n, rs, w);
          if (lane_ok[l]) SplitEnum::expand_edges(rt, rs, got.data(), w);
        });
      for (auto& th : lanes) th.join();
      for (int l = 0; l < 32; ++l) REQUIRE(lane_ok[l] == cpu);
      if (!cpu) continue;
      ++refined;
      lean.edges(acc, want.data());
      REQUIRE(got == want);
    }
    REQUIRE(refined > 200);
  }
}
