#include "benchmark.cuh"

#include <iomanip>
#include <iostream>

using namespace std;

void print_report(const BenchmarkReport& report) {
  cout << fixed << setprecision(3);
  cout << "=== " << report.name << " ===\n";
  for (const TimingResult& t : report.timings) {
    cout << "  " << t.name << ":  " << t.kernel_ms << " ms";
    if (t.total_ms != t.kernel_ms) {
      cout << "  (total: " << t.total_ms << " ms)";
    }
    cout << '\n';
  }
  cout << fixed << setprecision(2);
  for (const ComparisonResult& c : report.comparisons) {
    cout << "  Speedup (" << c.baseline << " -> " << c.target << "):  " << c.speedup << "x";
    if (c.correctness_checked) {
      cout << "   correct: " << (c.correct ? "yes" : "no");
    }
    cout << '\n';
  }
  cout << '\n';
}
