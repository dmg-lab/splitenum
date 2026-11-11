#pragma once

#include "enumerate.h"

namespace SplitEnum {
int parse_cli(int argc, char** argv, Options& opt);

void report(const RunStats& s, const Options& opt);
}
