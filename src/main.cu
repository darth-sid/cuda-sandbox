#include "benchmark.cuh"

int main() {
  BenchmarkResult vector_add_result = run_vector_add_benchmark(1 << 20, 256);
  print_result(vector_add_result);

  BenchmarkResult matmul_result = run_matmul_benchmark(512, 512, 512, 16);
  print_result(matmul_result);

  return (vector_add_result.correct && matmul_result.correct) ? 0 : 1;
}
