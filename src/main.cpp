#include <cstdio>
#include <exception>

#include "cli.h"
#include "tcp.h"

int main(int argc, char** argv) {
  SplitEnum::Options opt;
  const int rc = SplitEnum::parse_cli(argc, argv, opt);
  if (rc >= 0) return rc;
  try {
    if (!opt.connect.empty()) {
      SplitEnum::run_worker(opt);
      return 0;
    }
    if (opt.listen_port >= 0) {
      SplitEnum::TcpRunner runner(opt.out, opt.listen_port);
      const SplitEnum::RunStats s = SplitEnum::run(opt, &runner);
      if (!s.stopped) runner.finish();
      SplitEnum::report(s, opt);
      return 0;
    }
    SplitEnum::report(SplitEnum::run(opt), opt);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "splitenum: %s\n", e.what());
    return 1;
  }
}
