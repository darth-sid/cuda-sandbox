You are writing a small CUDA C++ project that benchmarks GPU kernels.

Goal:
Create a minimal but well-structured benchmarking harness for CUDA kernels, plus one trivial kernel (vector addition) to test the harness.

Constraints:
- Language: CUDA C++ (nvcc)
- Code should prioritize simplicity, readability, and modular structure.
- Avoid unnecessary abstractions.
- Every component should compile and run independently.
- Use CUDA events to measure kernel execution time.
- Include a CPU baseline for correctness and speed comparison.

Project structure to generate:

cuda-kernel-bench/
  Makefile
  include/
    cuda_check.cuh
    benchmark.cuh
    tensor_init.cuh
  src/
    main.cu
    benchmark.cu
    vector_add.cu
  README.md

Implementation requirements:

1. CUDA error handling
Create a macro CUDA_CHECK that wraps CUDA calls and throws/prints errors.

2. Benchmark harness
Implement a reusable benchmarking module that measures:

- CPU runtime
- host→device memcpy time
- kernel execution time
- device→host memcpy time
- end-to-end runtime

Use CUDA events for GPU timing.

Create a struct:

struct BenchmarkResult {
  std::string name;
  float cpu_ms;
  float h2d_ms;
  float kernel_ms;
  float d2h_ms;
  float end_to_end_ms;
  bool correct;
};

Provide helper functions:

time_cpu_ms(fn, iterations)
time_cuda_ms(fn, iterations)
print_result(BenchmarkResult)

3. Tensor utilities
Implement helpers:

fill_random(vector<float>&)
allclose(vector<float>& a, vector<float>& b)

Used for initializing inputs and verifying correctness.

4. Vector add kernel

Implement a simple CUDA kernel:

__global__ void vector_add_kernel(float* a, float* b, float* c, int n)

Each thread computes one element.

Also implement:

vector_add_cpu()

5. Benchmark function

Implement:

BenchmarkResult run_vector_add_benchmark(int n, int block_size)

The function should:

- allocate host vectors
- allocate device memory
- copy host → device
- run CPU baseline
- run GPU kernel
- copy device → host
- check correctness
- measure all timing metrics

6. main.cu

Main should run:

run_vector_add_benchmark(1<<20, 256)

and print the result.

7. Makefile

Provide a simple Makefile using nvcc:

targets:
- build
- run
- clean

Compile with:

nvcc -O3 -std=c++17

8. README

Explain:

- what the benchmark harness measures
- how to build
- how to run
- where new kernels should be added later

Also mention that this repo will later include kernels like:

- matrix multiplication
- reduction
- softmax

The code should compile and run on a CUDA-enabled machine or Google Colab.

Focus on correctness and clarity rather than heavy optimization.
