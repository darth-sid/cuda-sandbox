#include "cuda_check.cuh"

#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

void launch_vector_add(const float* d_a, const float* d_b, float* d_c, int n,
                       int block_size, cudaStream_t stream) {
  int grid_size = (n + block_size - 1) / block_size;
  vector_add_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_c, n);
}
