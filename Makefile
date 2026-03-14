NVCC := nvcc
NVCCFLAGS := -O3 -std=c++17 -Iinclude
TARGET := cuda_bench
SRC := src/main.cu src/benchmark.cu src/vector_add.cu

.PHONY: build run clean

build: $(TARGET)

$(TARGET): $(SRC) include/benchmark.cuh include/cuda_check.cuh include/tensor_init.cuh
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

run: build
	./$(TARGET)

clean:
	rm -f $(TARGET)
