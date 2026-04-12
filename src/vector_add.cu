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

void validate_vector_add_dims(int n, int block_size) {
  if (block_size <= 0) {
    cerr << "vector_add_kernel requires block_size > 0. Got block_size=" << block_size << endl;
    exit(EXIT_FAILURE);
  }

  if ((n % block_size) == 0) {
    return;
  }

  cerr << "vector_add_kernel requires n to be a multiple of block_size. Got n=" << n
       << ", block_size=" << block_size << endl;
  exit(EXIT_FAILURE);
}

void vector_add_cpu(const vector<float>& a, const vector<float>& b,
                    vector<float>& c) {
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    c[i] = a[i] + b[i];
  }
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

}  // namespace

void launch_vector_add(const float* d_a, const float* d_b, float* d_c, int n,
                       int block_size) {
  validate_vector_add_dims(n, block_size);
  int grid_size = n / block_size;
  vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaGetLastError());
}

BenchmarkResult run_vector_add_benchmark(int n, int block_size,
                                         const vector<NamedVectorAddLaunch>& kernels) {
  vector<float> h_a(n);
  vector<float> h_b(n);
  vector<float> h_cpu(n);

  fill_random(h_a);
  fill_random(h_b);
  float cpu_ms = time_cpu_ms([&]() { vector_add_cpu(h_a, h_b, h_cpu); }, kCpuIterations);

  DeviceBuffer<float> d_a(n);
  DeviceBuffer<float> d_b(n);
  DeviceBuffer<float> d_c(n);
  int bytes = n * sizeof(float);

  float h2d_ms = time_cuda_ms(
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), bytes, cudaMemcpyHostToDevice));
      },
      kTransferIterations);

  vector<float> h_gpu(n, 0.0f);
  float d2h_ms = time_cuda_ms(
      [&]() {
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));
      },
      kTransferIterations);

  vector<KernelBenchmarkResult> results;
  results.reserve(kernels.size());

  for (const NamedVectorAddLaunch& kernel : kernels) {
    float end_to_end_ms = time_cuda_ms(
        [&]() {
          CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), bytes, cudaMemcpyHostToDevice));
          kernel.launch(d_a.get(), d_b.get(), d_c.get(), n, block_size);
          CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));
        },
        kEndToEndIterations);

    float kernel_ms =
        time_cuda_ms([&]() { kernel.launch(d_a.get(), d_b.get(), d_c.get(), n, block_size); },
                     kKernelIterations);

    kernel.launch(d_a.get(), d_b.get(), d_c.get(), n, block_size);
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));

    results.push_back(KernelBenchmarkResult{
        kernel.name,
        kernel_ms,
        end_to_end_ms,
        allclose(h_cpu, h_gpu),
    });
  }

  return BenchmarkResult{
      "vector_add",
      cpu_ms,
      h2d_ms,
      d2h_ms,
      results,
  };
}
