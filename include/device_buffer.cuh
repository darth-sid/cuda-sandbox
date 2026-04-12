#ifndef DEVICE_BUFFER_CUH
#define DEVICE_BUFFER_CUH

#include "cuda_check.cuh"

#include <cstddef>
#include <cuda_runtime.h>

using namespace std;

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        cudaFree(ptr_);
      }

      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }

    return *this;
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

#endif
