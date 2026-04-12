#include "benchmark.cuh"
#include "benchmark_utils.cuh"
#include "cuda_check.cuh"
#include "device_buffer.cuh"
#include "tensor_init.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

namespace {

constexpr int NAIVE_BLOCK_SIZE = 16;
constexpr int TILE_SIZE = 32;

void matmul_cpu(const vector<float>& a, const vector<float>& b,
                vector<float>& c, int m, int n, int k) {
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

void validate_matmul_dims(const char* kernel_name, int m, int n, int k, int block_size,
                          bool require_k_multiple) {
  bool invalid_m = (m % block_size) != 0;
  bool invalid_n = (n % block_size) != 0;
  bool invalid_k = require_k_multiple && ((k % block_size) != 0);
  if (!invalid_m && !invalid_n && !invalid_k) {
    return;
  }

  cerr << kernel_name << " requires dimensions to be multiples of " << block_size;
  if (require_k_multiple) {
    cerr << " for m, n, and k";
  } else {
    cerr << " for m and n";
  }
  cerr << ". Got m=" << m << ", n=" << n << ", k=" << k << endl;
  exit(EXIT_FAILURE);
}

__global__ void naive_matmul_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    sum += a[row * k + i] * b[i * n + col];
  }
  c[row * n + col] = sum;
}

__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + ty;
  int col = blockIdx.x * TILE_SIZE + tx;
  float sum = 0.0f;
  for (int tile = 0; tile < k / TILE_SIZE; ++tile) {
    tile_a[ty][tx] = a[row * k + tile * TILE_SIZE + tx];
    tile_b[ty][tx] = b[(tile * TILE_SIZE + ty) * n + col];
    __syncthreads();

    for (int i = 0; i < TILE_SIZE; ++i) {
      sum += tile_a[ty][i] * tile_b[i][tx];
    }
    __syncthreads();
  }
  c[row * n + col] = sum;
}

}  // namespace

void launch_naive_matmul(const float* d_a, const float* d_b, float* d_c, int m, int n, int k) {
  validate_matmul_dims("naive_matmul_kernel", m, n, k, NAIVE_BLOCK_SIZE, false);
  dim3 block(NAIVE_BLOCK_SIZE, NAIVE_BLOCK_SIZE);
  dim3 grid(n / block.x, m / block.y);

  naive_matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
}

void launch_matmul(const float* d_a, const float* d_b, float* d_c, int m, int n, int k) {
  validate_matmul_dims("matmul_kernel", m, n, k, TILE_SIZE, true);
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid(n / block.x, m / block.y);

  matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
}

BenchmarkResult run_matmul_benchmark(int m, int n, int k,
                                     const vector<NamedMatmulLaunch>& kernels) {
  vector<float> h_a(m * k);
  vector<float> h_b(k * n);
  vector<float> h_cpu(m * n);

  fill_random(h_a);
  fill_random(h_b);
  float cpu_ms = time_cpu_ms([&]() { matmul_cpu(h_a, h_b, h_cpu, m, n, k); }, kCpuIterations);

  DeviceBuffer<float> d_a(h_a.size());
  DeviceBuffer<float> d_b(h_b.size());
  DeviceBuffer<float> d_c(h_cpu.size());

  int a_bytes = h_a.size() * sizeof(float);
  int b_bytes = h_b.size() * sizeof(float);
  int c_bytes = h_cpu.size() * sizeof(float);

  float h2d_ms = time_cuda_ms(
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), a_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), b_bytes, cudaMemcpyHostToDevice));
      },
      kTransferIterations);

  vector<float> h_gpu(m * n, 0.0f);
  float d2h_ms = time_cuda_ms(
      [&]() {
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), c_bytes, cudaMemcpyDeviceToHost));
      },
      kTransferIterations);

  vector<KernelBenchmarkResult> results;
  results.reserve(kernels.size());

  for (const NamedMatmulLaunch& kernel : kernels) {
    float end_to_end_ms = time_cuda_ms(
        [&]() {
          CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), a_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), b_bytes, cudaMemcpyHostToDevice));
          kernel.launch(d_a.get(), d_b.get(), d_c.get(), m, n, k);
          CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), c_bytes, cudaMemcpyDeviceToHost));
        },
        kEndToEndIterations);

    float kernel_ms =
        time_cuda_ms([&]() { kernel.launch(d_a.get(), d_b.get(), d_c.get(), m, n, k); },
                     kKernelIterations);

    kernel.launch(d_a.get(), d_b.get(), d_c.get(), m, n, k);
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), c_bytes, cudaMemcpyDeviceToHost));

    results.push_back(KernelBenchmarkResult{
        kernel.name,
        kernel_ms,
        end_to_end_ms,
        allclose(h_cpu, h_gpu, 1e-4f, 1e-3f),
    });
  }

  return BenchmarkResult{
      "matmul",
      cpu_ms,
      h2d_ms,
      d2h_ms,
      results,
  };
}
