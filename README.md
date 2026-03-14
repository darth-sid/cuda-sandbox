# CUDA Kernel Benchmarks

This project provides a minimal CUDA C++ benchmarking harness for GPU kernels and includes a vector addition example to validate the workflow.

The benchmark harness measures:

- CPU runtime for a reference implementation
- host to device transfer time
- kernel execution time
- device to host transfer time
- end-to-end GPU runtime

## Build

Run:

```bash
make build
```

## Run

Run:

```bash
make run
```

The default executable launches `run_vector_add_benchmark(1 << 20, 256)` and prints timing results plus a correctness check.

## Project layout

- `include/cuda_check.cuh`: CUDA error handling macro
- `include/benchmark.cuh`: benchmark result type and public API
- `include/tensor_init.cuh`: host-side initialization and verification helpers
- `src/benchmark.cu`: timing helpers and vector add benchmark flow
- `src/vector_add.cu`: CUDA vector addition kernel
- `src/main.cu`: entry point

Add new kernels in `src/` and expose their benchmark entry points through `include/benchmark.cuh` or a dedicated header as the project grows.

Planned future kernels include:

- matrix multiplication
- reduction
- softmax

This code is intended to compile on a CUDA-enabled machine or in Google Colab with CUDA configured.
