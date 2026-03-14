#include "benchmark.cuh"

int main() {
  BenchmarkResult result = run_vector_add_benchmark(1 << 20, 256);
  print_result(result);
  return result.correct ? 0 : 1;
}
