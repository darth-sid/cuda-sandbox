#include "benchmark.cuh"
#include "benchmark_utils.cuh"
#include "cuda_check.cuh"
#include "device_buffer.cuh"
#include "tensor_init.cuh"

#include <cuda_runtime.h>

#include <vector>

namespace {

void vector_add_cpu(const std::vector<float>& a, const std::vector<float>& b,
                    std::vector<float>& c) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    c[i] = a[i] + b[i];
  }
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

  DeviceBuffer<float> d_a(n);
  DeviceBuffer<float> d_b(n);
  DeviceBuffer<float> d_c(n);
  std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  float end_to_end_ms = time_cuda_ms(
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), bytes, cudaMemcpyHostToDevice));
        launch_vector_add(d_a.get(), d_b.get(), d_c.get(), n, block_size);
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));
      },
      kEndToEndIterations);

  float h2d_ms = time_cuda_ms(
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), bytes, cudaMemcpyHostToDevice));
      },
      kTransferIterations);

  float kernel_ms = time_cuda_ms(
      [&]() { launch_vector_add(d_a.get(), d_b.get(), d_c.get(), n, block_size); },
      kKernelIterations);

  float d2h_ms = time_cuda_ms(
      [&]() {
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));
      },
      kTransferIterations);

  float cpu_ms = time_cpu_ms([&]() { vector_add_cpu(h_a, h_b, h_cpu); }, kCpuIterations);

  launch_vector_add(d_a.get(), d_b.get(), d_c.get(), n, block_size);
  CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));
  bool correct = allclose(h_cpu, h_gpu);

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
