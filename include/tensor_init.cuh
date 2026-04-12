#ifndef TENSOR_INIT_CUH
#define TENSOR_INIT_CUH

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace std;

inline void fill_random(vector<float>& values) {
  static mt19937 rng(12345);
  static uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (float& value : values) {
    value = dist(rng);
  }
}

inline bool allclose(const vector<float>& a, const vector<float>& b,
                     float atol = 1e-5f, float rtol = 1e-4f) {
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); ++i) {
    float diff = fabs(a[i] - b[i]);
    float limit = atol + rtol * max(fabs(a[i]), fabs(b[i]));
    if (diff > limit) {
      return false;
    }
  }

  return true;
}

#endif
