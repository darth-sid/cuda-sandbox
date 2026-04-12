#ifndef CUDA_CHECK_CUH
#define CUDA_CHECK_CUH

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

using namespace std;

inline void cuda_check_impl(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    cerr << "CUDA error at " << file << ":" << line << " - "
         << cudaGetErrorString(error) << endl;
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(call) cuda_check_impl((call), __FILE__, __LINE__)

#endif
