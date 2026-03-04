#include <developmental-features/quantization/ProductQuantization.h>
#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Index.h>
#include <flatnav/quantization/ScalarQuantizedDistance.h>
#include <flatnav/util/Datatype.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "cnpy.h"

using flatnav::Index;
using flatnav::distances::DistanceInterface;
using flatnav::distances::InnerProductDistance;
using flatnav::distances::SquaredL2Distance;
using flatnav::quantization::ProductQuantizer;
using flatnav::quantization::ScalarQuantizedDistance;
using flatnav::util::DataType;

template <typename dist_t>
void buildIndex(float* data, std::unique_ptr<DistanceInterface<dist_t>> distance, int N, int M, int dim,
                int ef_construction, int build_num_threads, const std::string& save_file) {

  auto index = new Index<dist_t, int>(
      /* dist = */ std::move(distance), /* dataset_size = */ N,
      /* max_edges = */ M);

  index->setNumThreads(build_num_threads);

  auto start = std::chrono::high_resolution_clock::now();

  std::vector<int> labels(N);
  std::iota(labels.begin(), labels.end(), 0);
  index->template addBatch<float>(/* data = */ (void*)data,
                                  /* labels = */ labels,
                                  /* ef_construction */ ef_construction);

  auto stop = std::chrono::high_resolution_clock ::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::clog << "Build time: " << (float)duration.count() << " milliseconds" << std::endl;

  std::clog << "Saving index to: " << save_file << std::endl;
  index->saveIndex(/* filename = */ save_file);

  delete index;
}

void run(float* data, flatnav::distances::MetricType metric_type, int N, int M, int dim, int ef_construction,
         int build_num_threads, const std::string& save_file, int quantize = 0,
         size_t max_train_samples = 0) {

  if (quantize == 1) {
    // Product quantization. Parameters M and nbits should be adjusted accordingly.
    auto quantizer = std::make_unique<ProductQuantizer>(
        /* dim = */ dim, /* M = */ 8, /* nbits = */ 8,
        /* metric_type = */ metric_type);

    auto start = std::chrono::high_resolution_clock::now();
    quantizer->train(/* vectors = */ data, /* num_vectors = */ N);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::clog << "PQ training time: " << (float)duration.count() << " milliseconds" << std::endl;

    buildIndex<ProductQuantizer>(data, std::move(quantizer), N, M, dim, ef_construction, build_num_threads,
                                 save_file);

  } else if (quantize == 2) {
    // Scalar quantization (8-bit per dimension).
    if (metric_type == flatnav::distances::MetricType::L2) {
      auto sq = ScalarQuantizedDistance<flatnav::distances::MetricType::L2>::create(dim);

      auto start = std::chrono::high_resolution_clock::now();
      sq->train(data, static_cast<size_t>(N), max_train_samples);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      std::clog << "SQ training time: " << (float)duration.count() << " milliseconds" << std::endl;

      buildIndex<ScalarQuantizedDistance<flatnav::distances::MetricType::L2>>(
          data, std::move(sq), N, M, dim, ef_construction, build_num_threads, save_file);

    } else if (metric_type == flatnav::distances::MetricType::IP) {
      auto sq = ScalarQuantizedDistance<flatnav::distances::MetricType::IP>::create(dim);

      auto start = std::chrono::high_resolution_clock::now();
      sq->train(data, static_cast<size_t>(N), max_train_samples);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      std::clog << "SQ training time: " << (float)duration.count() << " milliseconds" << std::endl;

      buildIndex<ScalarQuantizedDistance<flatnav::distances::MetricType::IP>>(
          data, std::move(sq), N, M, dim, ef_construction, build_num_threads, save_file);
    }

  } else {
    // No quantization (quantize == 0).
    if (metric_type == flatnav::distances::MetricType::L2) {
      auto distance = SquaredL2Distance<>::create(dim);
      buildIndex<SquaredL2Distance<DataType::float32>>(data, std::move(distance), N, M, dim, ef_construction,
                                                       build_num_threads, save_file);

    } else if (metric_type == flatnav::distances::MetricType::IP) {
      auto distance = InnerProductDistance<>::create(dim);
      buildIndex<InnerProductDistance<DataType::float32>>(data, std::move(distance), N, M, dim,
                                                          ef_construction, build_num_threads, save_file);
    }
  }
}

int main(int argc, char** argv) {

  if (argc < 8) {
    std::clog << "Usage: " << std::endl;
    std::clog << "construct <quantize> <metric> <data> <M> <ef_construction> "
                 "<build_num_threads> <outfile> [max_train_samples]"
              << std::endl;
    std::clog
        << "\t <quantize> int, 0 for no quantization, 1 for product quantization, 2 for scalar quantization"
        << std::endl;
    std::clog << "\t <metric> int, 0 for L2, 1 for inner product (angular)" << std::endl;
    std::clog << "\t <data> npy file from ann-benchmarks" << std::endl;
    std::clog << "\t <M>: int " << std::endl;
    std::clog << "\t <ef_construction>: int " << std::endl;
    std::clog << "\t <build_num_threads>: int " << std::endl;
    std::clog << "\t <outfile>: where to stash the index" << std::endl;
    std::clog << "\t [max_train_samples]: int, optional, max vectors to sample for SQ training (0 = use all, "
                 "default 0)"
              << std::endl;

    return -1;
  }

  int quantize = std::stoi(argv[1]);
  int metric_id = std::stoi(argv[2]);
  cnpy::NpyArray datafile = cnpy::npy_load(argv[3]);
  int M = std::stoi(argv[4]);
  int ef_construction = std::stoi(argv[5]);

  if ((datafile.shape.size() != 2)) {
    return -1;
  }

  int dim = datafile.shape[1];
  int N = datafile.shape[0];

  std::clog << "Loading " << dim << "-dimensional dataset with N = " << N << std::endl;
  float* data = datafile.data<float>();
  flatnav::distances::MetricType metric_type =
      metric_id == 0 ? flatnav::distances::MetricType::L2 : flatnav::distances::MetricType::IP;

  size_t max_train_samples = (argc >= 9) ? std::stoull(argv[8]) : 0;

  run(/* data = */ data,
      /* metric_type = */ metric_type,
      /* N = */ N, /* M = */ M, /* dim = */ dim,
      /* ef_construction = */ ef_construction,
      /* build_num_threads = */ std::stoi(argv[6]),
      /* save_file = */ argv[7],
      /* quantize = */ quantize,
      /* max_train_samples = */ max_train_samples);

  return 0;
}