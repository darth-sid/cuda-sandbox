# CUDA Kernel Benchmarks

This project is a small sandbox for benchmarking CUDA kernels without a heavy framework.

The benchmark flow measures:

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

If `nvcc` is not on `PATH`, the `Makefile` also falls back to Colab's default CUDA compiler path: `/usr/local/cuda/bin/nvcc`.

## Run

Run:

```bash
make run
```

The default executable launches:

- `run_vector_add_benchmark(1 << 20, 256, {{"vector_add", launch_vector_add}})`
- `run_matmul_benchmark(512, 512, 512, {{"matmul_naive", launch_naive_matmul}, {"matmul", launch_matmul}})`

and prints timing results plus a correctness check for both.

## Google Colab

1. Open a new Colab notebook.
2. Set the runtime to GPU:
   `Runtime` -> `Change runtime type` -> `T4 GPU` or any available NVIDIA GPU.
3. Upload this repo or clone it into the notebook VM.

If the repo is uploaded as a zip:

```bash
!unzip cuda-kernel-benchmarks.zip
%cd cuda-kernel-benchmarks
```

If the repo is hosted in Git:

```bash
!git clone <your-repo-url>
%cd cuda-kernel-benchmarks
```

Then verify the CUDA toolchain and run the benchmark:

```bash
!nvidia-smi
!make print-env
!make run
```

If Colab starts without a GPU runtime, `nvidia-smi` will fail and the benchmark will not build correctly.

## Project layout

- `include/cuda_check.cuh`: CUDA error handling macro
- `include/benchmark.cuh`: benchmark result type and public API
- `include/tensor_init.cuh`: host-side initialization and verification helpers
- `src/benchmark.cu`: result formatting
- `src/vector_add.cu`: vector add kernel and the concrete benchmark flow
- `src/matmul.cu`: naive matrix multiplication kernel and matching benchmark flow
- `src/main.cu`: entry point

To add another kernel later, follow the same pattern as `src/vector_add.cu` and `src/matmul.cu`: keep the kernel, CPU reference implementation, and benchmark flow together unless shared code becomes clearly reusable. Planned additions include reduction and softmax.

This code is intended to compile on a CUDA-enabled Linux machine or Google Colab with an NVIDIA GPU runtime.
