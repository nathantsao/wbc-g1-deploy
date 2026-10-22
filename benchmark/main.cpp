#include "backends/backend.h"
#include "benchmark.h"
#include "options.h"
#include "report.h"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    const auto options = policy_benchmark::parse_options(argc, argv);
    if (options.list_devices) {
      policy_benchmark::list_devices(options, std::cout);
      return 0;
    }
    const auto result = policy_benchmark::run_benchmark(options);
    policy_benchmark::print_report(options, result, std::cout);
    if (!options.csv_path.empty()) {
      policy_benchmark::write_csv(options.csv_path, result.latency_us);
      std::cout << "latency_csv=" << options.csv_path << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark error: " << error.what() << '\n';
    return 1;
  }
}
