#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace {

void validate_vector_add_dims(int n, int block_size) {
  if (block_size <= 0) {
    std::cerr << "vector_add_kernel requires block_size > 0. Got block_size=" << block_size
              << std::endl;
    exit(EXIT_FAILURE);
  }

  if ((n % block_size) == 0) {
    return;
  }

  std::cerr << "vector_add_kernel requires n to be a multiple of block_size. Got n=" << n
            << ", block_size=" << block_size << std::endl;
  exit(EXIT_FAILURE);
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

}  // namespace

void launch_vector_add(const float* d_a, const float* d_b, float* d_c, int n, int block_size) {
  validate_vector_add_dims(n, block_size);
  int grid_size = n / block_size;
  vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaGetLastError());
}
