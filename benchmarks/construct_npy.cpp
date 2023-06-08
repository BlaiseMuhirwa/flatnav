#include "../flatnav/Index.h"
#include "../flatnav/distances/InnerProductDistance.h"
#include "../flatnav/distances/SquaredL2Distance.h"
#include "cnpy.h"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using flatnav::Index;
using flatnav::InnerProductDistance;
using flatnav::SquaredL2Distance;

struct BenchmarkFixture : public benchmark::fixture {
  std::unique_ptr<Index<SquaredL2Distance, int>> _index;
  int _N, _M, _dim, _ef_construction;

  void SetUp(benchmark::State &state) override {
    _N = state.range(0);
    _M = state.range(1);
    _dim = state.range(2);
    _ef_construction = state.range(3);

    auto distance = std::make_unique<SquaredL2Distance>(_dim);
    _index = std::make_unique<Index<SquaredL2Distance, int>>(
        /* dist = */ std::move(distance), /* dataset_size = */ _N,
        /* max_edges = */ _M);
  }

  void TearDown(benchmark::State &state) override { _index.reset(); }
};





template <typename dist_t>
void buildIndex(Index<dist_t, int> *index, float *data, int N, int M, int dim,
                int ef_construction) {
  for (int label = 0; label < N; label++) {
    float *element = data + (dim * label);
    index->add(/* data = */ (void *)element, /* label = */ label,
               /* ef_construction */ ef_construction);
  }
}

static void BM_buildL2Index(benchmark::State &state) {
  Index<SquaredL2Distance, int> *index = state.range(0);
  float *data = state.range(1);
  int N = state.range(2);
  int M = state.range(3);
  int dim = state.range(4);
  int ef_construction = state.range(5);
  for (auto _ : state) {
    buildIndex<SquaredL2Distance>(/* index = */ index, /* data = */ data,
                                  /* N = */ N, /* M = */ M, /* dim = */ dim,
                                  /* ef_construction = */ ef_construction);
  }
}

static void BM_buildIPIndex(benchmark::State &state) {
  Index<InnerProductDistance, int> index = state.range(0);
  float *data = state.range(1);
  int N = state.range(2);
  int M = state.range(3);
  int dim = state.range(4);
  int ef_construction = state.range(5);
  for (auto _ : state) {
    buildIndex<InnerProductDistance>(/* index = */ index, /* data = */ data,
                                     /* N = */ N, /* M = */ M, /* dim = */ dim,
                                     /* ef_construction = */ ef_construction);
  }
}

// template <typename dist_t>
// void run(float *data,
//          std::unique_ptr<flatnav::DistanceInterface<dist_t>> &&distance, int
//          N, int M, int dim, int ef_construction, const std::string
//          &save_file) {
//   auto index = new Index<dist_t, int>(
//       /* dist = */ std::move(distance), /* dataset_size = */ N,
//       /* max_edges = */ M);

//   BENCHMARK(BM_buildIndex);
//   BENCHMARK_MAIN();

// auto start = std::chrono::high_resolution_clock::now();

// for (int label = 0; label < N; label++) {
//   float *element = data + (dim * label);
//   index->add(/* data = */ (void *)element, /* label = */ label,
//              /* ef_construction */ ef_construction);
//   if (label % 100000 == 0)
//     std::clog << "." << std::flush;
// }
// std::clog << std::endl;

// auto stop = std::chrono::high_resolution_clock::now();
// auto duration =
//     std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
// std::clog << "Build time: " << (float)duration.count() << " milliseconds"
//           << std::endl;

// std::clog << "Saving index to: " << save_file << std::endl;
//   std::ofstream stream(save_file);
//   index->serialize(/* filename = */ stream);

//   delete index;
// }

int main(int argc, char **argv) {

  if (argc < 5) {
    std::clog << "Usage: " << std::endl;
    std::clog << "construct <metric> <data> <M> <ef_construction> <outfile>"
              << std::endl;
    std::clog << "\t <metric> int, 0 for L2, 1 for inner product (angular)"
              << std::endl;
    std::clog << "\t <data> npy file from ann-benchmarks" << std::endl;
    std::clog << "\t <M>: int " << std::endl;
    std::clog << "\t <ef_construction>: int " << std::endl;
    std::clog << "\t <outfile>: where to stash the index" << std::endl;

    return -1;
  }

  int metric_id = std::stoi(argv[1]);
  cnpy::NpyArray datafile = cnpy::npy_load(argv[2]);
  int M = std::stoi(argv[3]);
  int ef_construction = std::stoi(argv[4]);

  if ((datafile.shape.size() != 2)) {
    return -1;
  }

  int dim = datafile.shape[1];
  int N = datafile.shape[0];

  std::clog << "Loading " << dim << "-dimensional dataset with N = " << N
            << std::endl;
  float *data = datafile.data<float>();

  std::clog << "before index initialization " << std::endl;

  if (metric_id == 0) {
    auto distance = std::make_unique<SquaredL2Distance>(dim);
    auto index = new Index<SquaredL2Distance, int>(
        /* dist = */ std::move(distance), /* dataset_size = */ N,
        /* max_edges = */ M);

    // register benchmark
    BENCHMARK(BM_buildL2Index)->Args({index, data, N, M, dim, ef_construction});

    // run<SquaredL2Distance>(
    //     /* data = */ data,
    //     /* distance = */ std::move(distance),
    //     /* N = */ N, /* M = */ M, /* dim = */ dim,
    //     /* ef_construction = */ ef_construction, /* save_file = */ argv[5]);
  } else if (metric_id == 1) {
    auto distance = std::make_unique<InnerProductDistance>(dim);
    // run<InnerProductDistance>(
    //     /* data = */ data,
    //     /* distance = */ std::move(distance),
    //     /* N = */ N, /* M = */ M, dim,
    //     /* ef_construction = */ ef_construction, /* save_file = */ argv[5]);
    auto index = new Index<InnerProductDistance, int>(
        /* dist = */ std::move(distance), /* dataset_size = */ N,
        /* max_edges = */ M);

    BENCHMARK(BM_buildIPIndex)->Args({index, data, N, M, dim, ef_construction});

  } else {
    throw std::invalid_argument("Provided metric ID " +
                                std::to_string(metric_id) + "is invalid.");
  }
  benchmark::RunSpecifiedBenchmarks();

  return 0;
}
