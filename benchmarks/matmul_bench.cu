#include "benchmark.cuh"
#include "cuda_check.cuh"
#include "device_buffer.cuh"
#include "tensor_init.cuh"
#include "kernels/matmul.cuh"

#include <cuda_runtime.h>
#include <vector>

using namespace std;

BenchmarkReport run_matmul_bench(int m, int n, int k) {
  vector<float> h_a(static_cast<size_t>(m) * k);
  vector<float> h_b(static_cast<size_t>(k) * n);
  vector<float> h_cpu(static_cast<size_t>(m) * n);
  vector<float> h_gpu(static_cast<size_t>(m) * n, 0.0f);

  fill_random(h_a);
  fill_random(h_b);

  DeviceBuffer<float> d_a(h_a.size());
  DeviceBuffer<float> d_b(h_b.size());
  DeviceBuffer<float> d_c(static_cast<size_t>(m) * n);

  size_t a_bytes = h_a.size() * sizeof(float);
  size_t b_bytes = h_b.size() * sizeof(float);
  size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);

  CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), a_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), b_bytes, cudaMemcpyHostToDevice));

  BenchmarkSuite suite("matmul");

  suite.add_runner("naive_gpu", true,
      [&]() { launch_naive_matmul(d_a.get(), d_b.get(), d_c.get(), m, n, k); },
      [&]() -> vector<float> {
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), c_bytes, cudaMemcpyDeviceToHost));
        return h_gpu;
      },
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), a_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), b_bytes, cudaMemcpyHostToDevice));
        launch_naive_matmul(d_a.get(), d_b.get(), d_c.get(), m, n, k);
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), c_bytes, cudaMemcpyDeviceToHost));
      });

  suite.add_runner("tiled_gpu", true,
      [&]() { launch_matmul(d_a.get(), d_b.get(), d_c.get(), m, n, k); },
      [&]() -> vector<float> {
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), c_bytes, cudaMemcpyDeviceToHost));
        return h_gpu;
      },
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), a_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), b_bytes, cudaMemcpyHostToDevice));
        launch_matmul(d_a.get(), d_b.get(), d_c.get(), m, n, k);
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c.get(), c_bytes, cudaMemcpyDeviceToHost));
      });

  return suite.run("naive_gpu", 1e-4f, 1e-3f);
}
