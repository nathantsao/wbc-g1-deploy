#pragma once

#include "benchmark.h"
#include <iosfwd>

namespace policy_benchmark {

void print_report(const Options& options, const BenchmarkResult& result, std::ostream& out);
void write_csv(const std::string& path, const std::vector<double>& latency_us);

}  // namespace policy_benchmark
