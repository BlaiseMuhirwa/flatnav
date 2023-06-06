

#include <benchmark/benchmark.h>

void SomeFunction() {
    int total = 0;
    for (int i = 0; i < 100; i++) {
        benchmark::DoNotOptimize(total += i);
    }
}

static void BM_SomeFunction(benchmark::State& state) {
  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    SomeFunction();
  }
}
// Register the function as a benchmark
BENCHMARK(BM_SomeFunction);
// Run the benchmark
BENCHMARK_MAIN();