#include "benchmark.cuh"
#include "benchmark_utils.cuh"
#include "cuda_check.cuh"
#include "device_buffer.cuh"
#include "tensor_init.cuh"

#include <cuda_runtime.h>

#include <vector>

namespace {

void matmul_cpu(const std::vector<float>& a, const std::vector<float>& b,
                std::vector<float>& c, int m, int n, int k) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int inner = 0; inner < k; ++inner) {
        sum += a[row * k + inner] * b[inner * n + col];
      }
      c[row * n + col] = sum;
    }
  }
}

}  // namespace

__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int n,
                              int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float sum = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
      sum += a[row * k + inner] * b[inner * n + col];
    }
    c[row * n + col] = sum;
  }
}

namespace {

void launch_matmul(const float* d_a, const float* d_b, float* d_c, int m, int n, int k,
                   int block_size) {
  dim3 block(block_size, block_size);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

  matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

BenchmarkResult run_matmul_benchmark(int m, int n, int k, int block_size) {
  std::vector<float> h_a(static_cast<std::size_t>(m) * k);
  std::vector<float> h_b(static_cast<std::size_t>(k) * n);
  std::vector<float> h_cpu(static_cast<std::size_t>(m) * n);
  std::vector<float> h_gpu(static_cast<std::size_t>(m) * n, 0.0f);

  fill_random(h_a);
  fill_random(h_b);

  DeviceBuffer<float> d_a(h_a.size());
  DeviceBuffer<float> d_b(h_b.size());
  DeviceBuffer<float> d_c(h_cpu.size());

  std::size_t a_bytes = h_a.size() * sizeof(float);
  std::size_t b_bytes = h_b.size() * sizeof(float);
  std::size_t c_bytes = h_cpu.size() * sizeof(float);

  float end_to_end_ms = time_cuda_ms(
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), a_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), b_bytes, cudaMemcpyHostToDevice));
        launch_matmul(d_a.get(), d_b.get(), d_c.get(), m, n, k, block_size);
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), c_bytes, cudaMemcpyDeviceToHost));
      },
      kEndToEndIterations);

  float h2d_ms = time_cuda_ms(
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), a_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), b_bytes, cudaMemcpyHostToDevice));
      },
      kTransferIterations);

  float kernel_ms =
      time_cuda_ms([&]() { launch_matmul(d_a.get(), d_b.get(), d_c.get(), m, n, k, block_size); },
                   kKernelIterations);

  float d2h_ms = time_cuda_ms(
      [&]() {
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), c_bytes, cudaMemcpyDeviceToHost));
      },
      kTransferIterations);

  float cpu_ms = time_cpu_ms([&]() { matmul_cpu(h_a, h_b, h_cpu, m, n, k); }, kCpuIterations);

  launch_matmul(d_a.get(), d_b.get(), d_c.get(), m, n, k, block_size);
  CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), c_bytes, cudaMemcpyDeviceToHost));
  bool correct = allclose(h_cpu, h_gpu, 1e-4f, 1e-3f);

  return BenchmarkResult{
      "matmul_naive",
      cpu_ms,
      h2d_ms,
      kernel_ms,
      d2h_ms,
      end_to_end_ms,
      correct,
  };
}
