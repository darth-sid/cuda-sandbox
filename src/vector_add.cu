#include "benchmark.cuh"
#include "cuda_check.cuh"
#include "tensor_init.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <vector>

namespace {

constexpr int kEndToEndIterations = 1;
constexpr int kTransferIterations = 5;
constexpr int kKernelIterations = 20;
constexpr int kCpuIterations = 20;

void vector_add_cpu(const std::vector<float>& a, const std::vector<float>& b,
                    std::vector<float>& c) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    c[i] = a[i] + b[i];
  }
}

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

}  // namespace

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

namespace {

void launch_vector_add(const float* d_a, const float* d_b, float* d_c, int n,
                       int block_size) {
  int grid_size = (n + block_size - 1) / block_size;
  vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

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
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
        launch_vector_add(d_a, d_b, d_c, n, block_size);
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));
      },
      kEndToEndIterations);

  float h2d_ms = time_cuda_ms(
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
      },
      kTransferIterations);

  float kernel_ms = time_cuda_ms([&]() { launch_vector_add(d_a, d_b, d_c, n, block_size); },
                                 kKernelIterations);

  float d2h_ms = time_cuda_ms(
      [&]() {
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));
      },
      kTransferIterations);

  float cpu_ms = time_cpu_ms([&]() { vector_add_cpu(h_a, h_b, h_cpu); }, kCpuIterations);

  launch_vector_add(d_a, d_b, d_c, n, block_size);
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
