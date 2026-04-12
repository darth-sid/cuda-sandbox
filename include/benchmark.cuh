#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

#include <string>
#include <vector>

using namespace std;

template <typename LaunchFn>
struct NamedKernelLaunch {
  string name;
  LaunchFn launch;
};

struct KernelBenchmarkResult {
  string name;
  float kernel_ms;
  float end_to_end_ms;
  bool correct;
};

struct BenchmarkResult {
  string name;
  float cpu_ms;
  float h2d_ms;
  float d2h_ms;
  vector<KernelBenchmarkResult> kernels;
};

using VectorAddLaunchFn = void (*)(const float*, const float*, float*, int, int);
using MatmulLaunchFn = void (*)(const float*, const float*, float*, int, int, int);

using NamedVectorAddLaunch = NamedKernelLaunch<VectorAddLaunchFn>;
using NamedMatmulLaunch = NamedKernelLaunch<MatmulLaunchFn>;

void launch_vector_add(const float* d_a, const float* d_b, float* d_c, int n, int block_size);
void launch_naive_matmul(const float* d_a, const float* d_b, float* d_c, int m, int n, int k);
void launch_matmul(const float* d_a, const float* d_b, float* d_c, int m, int n, int k);

bool all_kernels_correct(const BenchmarkResult& result);
void print_result(const BenchmarkResult& result);
BenchmarkResult run_vector_add_benchmark(int n, int block_size,
                                         const vector<NamedVectorAddLaunch>& kernels);
BenchmarkResult run_matmul_benchmark(int m, int n, int k,
                                     const vector<NamedMatmulLaunch>& kernels);

#endif
