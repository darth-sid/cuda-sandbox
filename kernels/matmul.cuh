#pragma once

void launch_naive_matmul(const float* d_a, const float* d_b, float* d_c, int m, int n, int k);
void launch_matmul(const float* d_a, const float* d_b, float* d_c, int m, int n, int k);
