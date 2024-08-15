#include "cnpy.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Index.h>
#include <flatnav/util/Datatype.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <quantization/ProductQuantization.h>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using flatnav::Index;
using flatnav::distances::DistanceInterface;
using flatnav::distances::InnerProductDistance;
using flatnav::distances::SquaredL2Distance;
using flatnav::quantization::ProductQuantizer;
using flatnav::util::DataType;

template <typename dist_t>
void buildIndex(float *data,
                std::unique_ptr<DistanceInterface<dist_t>> distance, int N,
                int M, int dim, int ef_construction, int build_num_threads,
                const std::string &save_file) {

  std::cout << "[info] building index" << std::endl;
  auto index = new Index<dist_t, int>(
      /* dist = */ std::move(distance), /* dataset_size = */ N,
      /* max_edges = */ M);

  index->setNumThreads(build_num_threads);

  auto start = std::chrono::high_resolution_clock::now();

  std::vector<int> labels(N);
  std::iota(labels.begin(), labels.end(), 0);
  index->template addBatch<float>(/* data = */ (void *)data,
                                  /* labels = */ labels,
                                  /* ef_construction */ ef_construction);

  exit(0);

  auto stop = std::chrono::high_resolution_clock ::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::clog << "Build time: " << (float)duration.count() << " milliseconds"
            << std::endl;

  std::clog << "Saving index to: " << save_file << std::endl;
  index->saveIndex(/* filename = */ save_file);

  delete index;
}

void run(float *data, flatnav::distances::MetricType metric_type, int N, int M,
         int dim, int ef_construction, int build_num_threads,
         const std::string &save_file, bool quantize = false) {

  if (quantize) {
    // Parameters M and nbits should be adjusted accordingly.
    auto quantizer = std::make_unique<ProductQuantizer>(
        /* dim = */ dim, /* M = */ 8, /* nbits = */ 8,
        /* metric_type = */ metric_type);

    auto start = std::chrono::high_resolution_clock::now();
    quantizer->train(/* vectors = */ data, /* num_vectors = */ N);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::clog << "Quantization time: " << (float)duration.count()
              << " milliseconds" << std::endl;

    buildIndex<ProductQuantizer>(data, std::move(quantizer), N, M, dim,
                                 ef_construction, build_num_threads, save_file);

  } else {
    if (metric_type == flatnav::distances::MetricType::L2) {
      auto distance = SquaredL2Distance<>::create(dim);
      buildIndex<SquaredL2Distance<DataType::float32>>(
          data, std::move(distance), N, M, dim, ef_construction,
          build_num_threads, save_file);

    } else if (metric_type == flatnav::distances::MetricType::IP) {
      auto distance = InnerProductDistance<>::create(dim);
      buildIndex<InnerProductDistance<DataType::float32>>(
          data, std::move(distance), N, M, dim, ef_construction,
          build_num_threads, save_file);
    }
  }
}

int main(int argc, char **argv) {

  if (argc < 8) {
    std::clog << "Usage: " << std::endl;
    std::clog << "construct <quantize> <metric> <data> <M> <ef_construction> "
                 "<build_num_threads> <outfile>"
              << std::endl;
    std::clog << "\t <quantize> int, 0 for no quantization, 1 for quantization"
              << std::endl;
    std::clog << "\t <metric> int, 0 for L2, 1 for inner product (angular)"
              << std::endl;
    std::clog << "\t <data> npy file from ann-benchmarks" << std::endl;
    std::clog << "\t <M>: int " << std::endl;
    std::clog << "\t <ef_construction>: int " << std::endl;
    std::clog << "\t <build_num_threads>: int " << std::endl;
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
  flatnav::distances::MetricType metric_type =
      metric_id == 0 ? flatnav::distances::MetricType::L2
                     : flatnav::distances::MetricType::IP;

  run(/* data = */ data,
      /* metric_type = */ metric_type,
      /* N = */ N, /* M = */ M, /* dim = */ dim,
      /* ef_construction = */ ef_construction,
      /* build_num_threads = */ std::stoi(argv[6]),
      /* save_file = */ argv[7],
      /* quantize = */ quantize);

  return 0;
}