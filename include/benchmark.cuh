#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

#include <functional>
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

float time_cpu_ms(const std::function<void()>& fn, int iterations = 1);
float time_cuda_ms(const std::function<void(cudaStream_t)>& fn, int iterations = 1);
void print_result(const BenchmarkResult& result);
BenchmarkResult run_vector_add_benchmark(int n, int block_size);

#endif
