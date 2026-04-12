#include "benchmark.cuh"

int main() {
  BenchmarkResult vector_add_result =
      run_vector_add_benchmark(1 << 20, 256, {{"vector_add", launch_vector_add}});
  print_result(vector_add_result);

  BenchmarkResult matmul_result = run_matmul_benchmark(
      512, 512, 512, {{"matmul_naive", launch_naive_matmul}, {"matmul", launch_matmul}});
  print_result(matmul_result);

  return (all_kernels_correct(vector_add_result) && all_kernels_correct(matmul_result)) ? 0 : 1;
}
