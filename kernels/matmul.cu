#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace {

constexpr int NAIVE_BLOCK_SIZE = 32;
constexpr int TILE_SIZE = 32;

void validate_matmul_dims(const char* kernel_name, int m, int n, int k, int block_size,
                          bool require_k_multiple) {
  bool invalid_m = (m % block_size) != 0;
  bool invalid_n = (n % block_size) != 0;
  bool invalid_k = require_k_multiple && ((k % block_size) != 0);
  if (!invalid_m && !invalid_n && !invalid_k) {
    return;
  }

  std::cerr << kernel_name << " requires dimensions to be multiples of " << block_size;
  if (require_k_multiple) {
    std::cerr << " for m, n, and k";
  } else {
    std::cerr << " for m and n";
  }
  std::cerr << ". Got m=" << m << ", n=" << n << ", k=" << k << std::endl;
  exit(EXIT_FAILURE);
}

__global__ void naive_matmul_kernel(const float* a, const float* b, float* c,
                                    int m, int n, int k) {
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
