NVCC ?= $(shell command -v nvcc 2>/dev/null)
ifeq ($(NVCC),)
NVCC := /usr/local/cuda/bin/nvcc
endif

NVCCFLAGS := -O2 -std=c++17 -I. -Iinclude
TARGET := bench
SRC := main.cu benchmarks/vector_add_bench.cu benchmarks/matmul_bench.cu \
       kernels/vector_add.cu kernels/matmul.cu src/report.cu

.PHONY: build run clean dryrun print-env

print-env:
	@echo "NVCC=$(NVCC)"

build: $(TARGET)

$(TARGET): $(SRC) \
           include/benchmark.cuh include/timing.cuh \
           include/cuda_check.cuh include/device_buffer.cuh include/tensor_init.cuh \
           kernels/vector_add.cuh kernels/matmul.cuh \
           benchmarks/vector_add_bench.cuh benchmarks/matmul_bench.cuh
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

dryrun:
	$(NVCC) $(NVCCFLAGS) --dryrun $(SRC) -o $(TARGET) 2>&1 | head -20

run: build
	./$(TARGET)

bench_no_l1: $(SRC)
		$(NVCC) $(NVCCFLAGS) -Xptxas -dlcm=cg $(SRC) -o bench_no_l1

bench_no_l2: $(SRC)
		$(NVCC) $(NVCCFLAGS) -Xptxas -dlcm=cs $(SRC) -o bench_no_l1

clean:
	rm -f $(TARGET)
