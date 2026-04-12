#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

#include <string>

struct BenchmarkResult {
  std::string name;
  float cpu_ms;
  float h2d_ms;
  float kernel_ms;
  float d2h_ms;
  float end_to_end_ms;
  bool correct;
};

void print_result(const BenchmarkResult& result);
BenchmarkResult run_vector_add_benchmark(int n, int block_size);

#endif
