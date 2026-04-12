#include "benchmark.cuh"
#include "cuda_check.cuh"
#include "device_buffer.cuh"
#include "tensor_init.cuh"
#include "kernels/vector_add.cuh"

#include <cuda_runtime.h>
#include <vector>

using namespace std;

namespace {

void vector_add_cpu(const vector<float>& a, const vector<float>& b, vector<float>& c) {
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    c[i] = a[i] + b[i];
  }
}

}  // namespace

BenchmarkReport run_vector_add_bench(int n, int block_size) {
  vector<float> h_a(n);
  vector<float> h_b(n);
  vector<float> h_cpu(n);
  vector<float> h_gpu(n, 0.0f);

  fill_random(h_a);
  fill_random(h_b);

  DeviceBuffer<float> d_a(n);
  DeviceBuffer<float> d_b(n);
  DeviceBuffer<float> d_c(n);

  size_t bytes = static_cast<size_t>(n) * sizeof(float);

  CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), bytes, cudaMemcpyHostToDevice));

  BenchmarkSuite suite("vector_add");

  suite.add_runner("cpu", false,
      [&]() { vector_add_cpu(h_a, h_b, h_cpu); },
      [&]() -> vector<float> { return h_cpu; });

  suite.add_runner("gpu", true,
      [&]() { launch_vector_add(d_a.get(), d_b.get(), d_c.get(), n, block_size); },
      [&]() -> vector<float> {
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));
        return h_gpu;
      },
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), bytes, cudaMemcpyHostToDevice));
        launch_vector_add(d_a.get(), d_b.get(), d_c.get(), n, block_size);
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));
      });

  return suite.run("cpu");
}
