#include "benchmark.cuh"

#include <iomanip>
#include <iostream>

using namespace std;

bool all_kernels_correct(const BenchmarkResult& result) {
  for (const KernelBenchmarkResult& kernel : result.kernels) {
    if (!kernel.correct) {
      return false;
    }
  }
  return true;
}

void print_result(const BenchmarkResult& result) {
  cout << fixed << setprecision(3);
  cout << "Benchmark: " << result.name << '\n';
  cout << "  CPU:      " << result.cpu_ms << " ms\n";
  cout << "  H2D copy: " << result.h2d_ms << " ms\n";
  cout << "  D2H copy: " << result.d2h_ms << " ms\n";
  for (const KernelBenchmarkResult& kernel : result.kernels) {
    cout << "  Kernel [" << kernel.name << "]:\n";
    cout << "    Kernel:     " << kernel.kernel_ms << " ms\n";
    cout << "    End-to-end: " << kernel.end_to_end_ms << " ms\n";
    cout << "    Correct:    " << (kernel.correct ? "yes" : "no") << '\n';
  }
}
