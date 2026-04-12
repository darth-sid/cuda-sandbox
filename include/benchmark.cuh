#pragma once

#include "timing.cuh"
#include "tensor_init.cuh"

#include <cuda_runtime.h>
#include <functional>
#include <optional>
#include <string>
#include <vector>

struct TimingResult {
  std::string name;
  float kernel_ms;
  float total_ms;
};

struct ComparisonResult {
  std::string baseline;
  std::string target;
  float speedup;
  bool correctness_checked;
  bool correct;
};

struct BenchmarkReport {
  std::string name;
  std::vector<TimingResult> timings;
  std::vector<ComparisonResult> comparisons;
};

void print_report(const BenchmarkReport& report);

class BenchmarkSuite {
public:
  using RunFn    = std::function<void()>;
  using OutputFn = std::function<std::vector<float>()>;

  explicit BenchmarkSuite(std::string name) : name_(std::move(name)) {}

  void add_runner(std::string name, bool is_gpu, RunFn run_fn, OutputFn get_output_fn,
                  std::optional<RunFn> total_fn = std::nullopt)
  {
    runners_.push_back({std::move(name), is_gpu, std::move(run_fn),
                        std::move(get_output_fn), std::move(total_fn)});
  }

  BenchmarkReport run(const std::string& baseline = "",
                      float atol = 1e-5f, float rtol = 1e-4f)
  {
    BenchmarkReport report;
    report.name = name_;

    std::vector<std::vector<float>> outputs(runners_.size());

    for (size_t i = 0; i < runners_.size(); ++i) {
      const Runner& r = runners_[i];

      // Warmup
      r.run_fn();
      if (r.is_gpu) {
        CUDA_CHECK(cudaDeviceSynchronize());
      }
      if (r.total_fn.has_value()) {
        (*r.total_fn)();
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      // Time kernel
      float kernel_ms;
      if (r.is_gpu) {
        kernel_ms = time_cuda_ms(r.run_fn, kKernelIterations);
      } else {
        kernel_ms = time_cpu_ms(r.run_fn, kCpuIterations);
      }

      // Time total pipeline (if provided)
      float total_ms = kernel_ms;
      if (r.total_fn.has_value()) {
        total_ms = time_cuda_ms(*r.total_fn, kTotalIterations);
      }

      report.timings.push_back({r.name, kernel_ms, total_ms});

      // Capture output: re-run kernel so buffer holds this runner's result
      r.run_fn();
      if (r.is_gpu) {
        CUDA_CHECK(cudaDeviceSynchronize());
      }
      outputs[i] = r.get_output_fn();
    }

    if (baseline.empty()) {
      return report;
    }

    // Locate baseline runner
    int baseline_idx = -1;
    for (size_t i = 0; i < runners_.size(); ++i) {
      if (runners_[i].name == baseline) {
        baseline_idx = static_cast<int>(i);
        break;
      }
    }
    if (baseline_idx < 0) {
      return report;
    }

    const std::vector<float>& baseline_output = outputs[baseline_idx];
    float baseline_kernel_ms = report.timings[baseline_idx].kernel_ms;

    for (size_t i = 0; i < runners_.size(); ++i) {
      if (static_cast<int>(i) == baseline_idx) continue;

      float speedup = baseline_kernel_ms / report.timings[i].kernel_ms;
      bool correct  = allclose(baseline_output, outputs[i], atol, rtol);
      report.comparisons.push_back({baseline, runners_[i].name, speedup, true, correct});
    }

    return report;
  }

private:
  struct Runner {
    std::string name;
    bool is_gpu;
    RunFn run_fn;
    OutputFn get_output_fn;
    std::optional<RunFn> total_fn;
  };

  std::string name_;
  std::vector<Runner> runners_;
};
