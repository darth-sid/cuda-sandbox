#include "benchmark.cuh"
#include "benchmarks/matmul_bench.cuh"
#include "benchmarks/vector_add_bench.cuh"

int main() {
  auto va_report = run_vector_add_bench(1 << 20, 256);
  print_report(va_report);

  auto mm_report = run_matmul_bench(512, 512, 512);
  print_report(mm_report);

  return 0;
}
