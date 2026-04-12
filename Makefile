NVCC ?= $(shell command -v nvcc 2>/dev/null)
ifeq ($(NVCC),)
NVCC := /usr/local/cuda/bin/nvcc
endif

NVCCFLAGS := -O3 -std=c++17 -Iinclude
TARGET := cuda_bench
SRC := src/main.cu src/benchmark.cu src/vector_add.cu src/matmul.cu

.PHONY: build run clean print-env

print-env:
	@echo "NVCC=$(NVCC)"

build: $(TARGET)

$(TARGET): $(SRC) include/benchmark.cuh include/benchmark_utils.cuh include/cuda_check.cuh include/device_buffer.cuh include/tensor_init.cuh
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

run: build
	./$(TARGET)

clean:
	rm -f $(TARGET)
