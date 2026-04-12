#include "benchmark.cuh"

#include <iomanip>
#include <iostream>
void print_result(const BenchmarkResult& result) {
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Benchmark: " << result.name << '\n';
  std::cout << "  CPU:        " << result.cpu_ms << " ms\n";
  std::cout << "  H2D copy:   " << result.h2d_ms << " ms\n";
  std::cout << "  Kernel:     " << result.kernel_ms << " ms\n";
  std::cout << "  D2H copy:   " << result.d2h_ms << " ms\n";
  std::cout << "  End-to-end: " << result.end_to_end_ms << " ms\n";
  std::cout << "  Correct:    " << (result.correct ? "yes" : "no") << '\n';
}
