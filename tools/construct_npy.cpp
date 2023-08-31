#include "cnpy.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <flatnav/Index.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <quantization/BaseProductQuantization.h>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using flatnav::DistanceInterface;
using flatnav::Index;
using flatnav::InnerProductDistance;
using flatnav::SquaredL2Distance;
using flatnav::quantization::ProductQuantizer;

template <typename dist_t>
void run(float *data, std::shared_ptr<DistanceInterface<dist_t>> &&distance,
         int N, int M, int dim, int ef_construction,
         const std::string &save_file, bool quantize = false) {

  std::unique_ptr<ProductQuantizer<dist_t>> pq = nullptr;
  if (quantize) {
    std::clog << "Quantizing data" << std::endl;

    pq = std::make_unique<ProductQuantizer<dist_t>>(
        /* dim = */ dim, /* M = */ 8, /* nbits = */ 16);

    auto start = std::chrono::high_resolution_clock::now();
    pq->train(/* vectors = */ data, /* num_vectors = */ N);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::clog << "Quantization time: " << (float)duration.count()
              << " milliseconds" << std::endl;
  }

  auto index = new Index<dist_t, int>(
      /* dist = */ distance, /* dataset_size = */ N,
      /* max_edges = */ M, /* pq = */ std::move(pq));

  auto start = std::chrono::high_resolution_clock::now();

  for (int label = 0; label < N; label++) {
    float *element = data + (dim * label);
    index->add(/* data = */ (void *)element, /* label = */ label,
               /* ef_construction */ ef_construction);
    if (label % 10000 == 0)
      std::clog << "." << std::flush;
  }
  std::clog << std::endl;

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::clog << "Build time: " << (float)duration.count() << " milliseconds"
            << std::endl;

  std::clog << "Saving index to: " << save_file << std::endl;
  index->saveIndex(/* filename = */ save_file);

  delete index;
}

int main(int argc, char **argv) {

  if (argc < 7) {
    std::clog << "Usage: " << std::endl;
    std::clog << "construct <quantize> <metric> <data> <M> <ef_construction> "
                 "<outfile>"
              << std::endl;
    std::clog << "\t <quantize> int, 0 for no quantization, 1 for quantization"
              << std::endl;
    std::clog << "\t <metric> int, 0 for L2, 1 for inner product (angular)"
              << std::endl;
    std::clog << "\t <data> npy file from ann-benchmarks" << std::endl;
    std::clog << "\t <M>: int " << std::endl;
    std::clog << "\t <ef_construction>: int " << std::endl;
    std::clog << "\t <outfile>: where to stash the index" << std::endl;

    return -1;
  }

  bool quantize = std::stoi(argv[1]) ? true : false;
  int metric_id = std::stoi(argv[2]);
  cnpy::NpyArray datafile = cnpy::npy_load(argv[3]);
  int M = std::stoi(argv[4]);
  int ef_construction = std::stoi(argv[5]);

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
    run<SquaredL2Distance>(
        /* data = */ data,
        /* distance = */ std::move(distance),
        /* N = */ N, /* M = */ M, /* dim = */ dim,
        /* ef_construction = */ ef_construction, /* save_file = */ argv[6],
        /* quantize = */ quantize);
  } else if (metric_id == 1) {
    auto distance = std::make_unique<InnerProductDistance>(dim);
    run<InnerProductDistance>(
        /* data = */ data,
        /* distance = */ std::move(distance),
        /* N = */ N, /* M = */ M, dim,
        /* ef_construction = */ ef_construction, /* save_file = */ argv[6],
        /* quantize = */ quantize);
  } else {
    throw std::invalid_argument("Provided metric ID " +
                                std::to_string(metric_id) + "is invalid.");
  }

  return 0;
}