# CUDA Kernel Benchmarks

Standalone CUDA benchmark harness for comparing kernel implementations against each other and against CPU baselines.

Each benchmark is a set of **runners** (CPU, naive GPU, optimized GPU) that all solve the same problem. Any runner can serve as the correctness baseline.

## Build

```bash
make build
```

If `nvcc` is not on `PATH`, the `Makefile` falls back to `/usr/local/cuda/bin/nvcc`.

## Run

```bash
make run
```

Expected output:

```
=== vector_add ===
  cpu:  X.XXX ms
  gpu:  X.XXX ms  (total: X.XXX ms)
  Speedup (cpu -> gpu):  XX.XXx   correct: yes

=== matmul ===
  cpu:  X.XXX ms
  naive_gpu:  X.XXX ms  (total: X.XXX ms)
  tiled_gpu:  X.XXX ms  (total: X.XXX ms)
  Speedup (cpu -> naive_gpu):  XX.XXx   correct: yes
  Speedup (cpu -> tiled_gpu):  XX.XXx   correct: yes
```

## Google Colab

1. Open a new Colab notebook.
2. Set runtime to GPU: `Runtime` → `Change runtime type` → `T4 GPU` (or any NVIDIA GPU).
3. Clone or upload this repo:

```bash
!git clone <your-repo-url>
%cd cuda-kernel-benchmarks
```

Then verify and run:

```bash
!nvidia-smi
!make print-env
!make run
```

## Project layout

```
include/
  cuda_check.cuh        CUDA_CHECK macro
  device_buffer.cuh     RAII DeviceBuffer<T>
  tensor_init.cuh       fill_random, allclose
  timing.cuh            time_cpu_ms, time_cuda_ms, iteration constants
  benchmark.cuh         BenchmarkSuite, TimingResult, ComparisonResult, BenchmarkReport

kernels/
  vector_add.{cuh,cu}   vector add kernel + launch_vector_add
  matmul.{cuh,cu}       matmul kernels + launch_naive_matmul, launch_matmul

benchmarks/
  vector_add_bench.{cuh,cu}   run_vector_add_bench (cpu + gpu runners)
  matmul_bench.{cuh,cu}       run_matmul_bench (cpu + naive_gpu + tiled_gpu runners)

src/
  report.cu             print_report implementation

main.cu                 entry point
```

## Adding a new kernel

1. Add `kernels/<name>.cuh` and `kernels/<name>.cu` with the kernel and launch function.
2. Add `benchmarks/<name>_bench.cuh` and `benchmarks/<name>_bench.cu` — create a `BenchmarkSuite`, register runners with `add_runner`, return `suite.run("cpu")`.
3. Include the new bench header in `main.cu` and call `print_report(run_<name>_bench(...))`.
4. Add the new `.cu` files to `SRC` in the `Makefile`.

Planned additions: reduction, softmax.
