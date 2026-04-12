#pragma once

#include "cuda_check.cuh"

#include <chrono>
#include <cuda_runtime.h>

constexpr int kKernelIterations = 20;
constexpr int kCpuIterations = 20;
constexpr int kTotalIterations = 5;

template <typename Operation>
float time_cpu_ms(Operation operation, int iterations) {
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    operation();
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<float, std::milli> elapsed = end - start;
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
