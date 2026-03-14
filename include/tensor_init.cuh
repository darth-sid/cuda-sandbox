#ifndef TENSOR_INIT_CUH
#define TENSOR_INIT_CUH

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

inline void fill_random(std::vector<float>& values) {
  static std::mt19937 rng(12345);
  static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (float& value : values) {
    value = dist(rng);
  }
}

inline bool allclose(const std::vector<float>& a, const std::vector<float>& b,
                     float atol = 1e-5f, float rtol = 1e-4f) {
  if (a.size() != b.size()) {
    return false;
  }

  for (std::size_t i = 0; i < a.size(); ++i) {
    float diff = std::fabs(a[i] - b[i]);
    float limit = atol + rtol * std::max(std::fabs(a[i]), std::fabs(b[i]));
    if (diff > limit) {
      return false;
    }
  }

  return true;
}

#endif
