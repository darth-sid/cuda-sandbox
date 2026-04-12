#ifndef BENCHMARK_UTILS_CUH
#define BENCHMARK_UTILS_CUH

#include "cuda_check.cuh"

#include <chrono>
#include <cuda_runtime.h>

using namespace std;

constexpr int kEndToEndIterations = 1;
constexpr int kTransferIterations = 5;
constexpr int kKernelIterations = 20;
constexpr int kCpuIterations = 20;

template <typename Operation>
float time_cpu_ms(Operation operation, int iterations) {
  auto start = chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    operation();
  }
  auto end = chrono::high_resolution_clock::now();

  chrono::duration<float, milli> elapsed = end - start;
  return elapsed.count() / static_cast<float>(iterations);
}

template <typename Operation>
float time_cuda_ms(Operation operation, int iterations) {
  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  float total_ms = 0.0f;
  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(cudaEventRecord(start_event));
    operation();
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    total_ms += elapsed_ms;
  }

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
  return total_ms / static_cast<float>(iterations);
}

#endif
