#include "benchmark.cuh"

#include "cuda_check.cuh"
#include "tensor_init.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void vector_add_cpu(const std::vector<float>& a, const std::vector<float>& b,
                    std::vector<float>& c) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    c[i] = a[i] + b[i];
  }
}

}  // namespace

void launch_vector_add(const float* d_a, const float* d_b, float* d_c, int n,
                       int block_size, cudaStream_t stream);

float time_cpu_ms(const std::function<void()>& fn, int iterations) {
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    fn();
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<float, std::milli> elapsed = end - start;
  return elapsed.count() / static_cast<float>(iterations);
}

float time_cuda_ms(const std::function<void(cudaStream_t)>& fn, int iterations) {
  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  float total_ms = 0.0f;
  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(cudaEventRecord(start_event));
    fn(0);
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

void print_result(const BenchmarkResult& result) {
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Benchmark: " << result.name << '\n';
  std::cout << "  CPU:        " << result.cpu_ms << " ms\n";
  std::cout << "  H2D copy:   " << result.h2d_ms << " ms\n";
  std::cout << "  Kernel:     " << result.kernel_ms << " ms\n";
  std::cout << "  D2H copy:   " << result.d2h_ms << " ms\n";
  std::cout << "  End-to-end: " << result.end_to_end_ms << " ms\n";
  std::cout << "  Correct:    " << (result.correct ? "yes" : "no") << '\n';
}

BenchmarkResult run_vector_add_benchmark(int n, int block_size) {
  std::vector<float> h_a(n);
  std::vector<float> h_b(n);
  std::vector<float> h_cpu(n);
  std::vector<float> h_gpu(n, 0.0f);

  fill_random(h_a);
  fill_random(h_b);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes));

  float end_to_end_ms = time_cuda_ms(
      [&](cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice, stream));

        launch_vector_add(d_a, d_b, d_c, n, block_size, stream);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(h_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream));
      },
      1);

  float h2d_ms = time_cuda_ms(
      [&](cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice, stream));
      },
      5);

  float kernel_ms = time_cuda_ms(
      [&](cudaStream_t stream) {
        launch_vector_add(d_a, d_b, d_c, n, block_size, stream);
        CUDA_CHECK(cudaGetLastError());
      },
      20);

  float d2h_ms = time_cuda_ms(
      [&](cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(h_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream));
      },
      5);

  float cpu_ms = time_cpu_ms([&]() { vector_add_cpu(h_a, h_b, h_cpu); }, 20);

  CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  bool correct = allclose(h_cpu, h_gpu);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return BenchmarkResult{
      "vector_add",
      cpu_ms,
      h2d_ms,
      kernel_ms,
      d2h_ms,
      end_to_end_ms,
      correct,
  };
}
