#include "cnpy.h"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <benchmarks/config_parser.h>
#include <chrono>
#include <cmath>
#include <flatnav/Index.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h> // for getcwd
#include <utility>
#include <vector>

#define GETCWD getcwd

using flatnav::Index;
using flatnav::InnerProductDistance;
using flatnav::SquaredL2Distance;

// It is not great that google-benchmark doesn't support passing strings
// directly as arguments to BENCHMARK_REGISTER_F()->Args({}). So, the following
// global map allows us to get dataset file paths given dataset unique IDs.
// Maybe there's a better way to do this, but haven't found it just yet.
std::unordered_map<uint32_t, std::string> dataset_id_to_filepath;

template <typename dist_t>
void buildIndex(std::shared_ptr<Index<dist_t, int>> index, float *data, int N,
                int M, int dim, int ef_construction) {
  for (int label = 0; label < N; label++) {
    float *element = data + (dim * label);
    index->add(/* data = */ (void *)element, /* label = */ label,
               /* ef_construction */ ef_construction);
  }
}

template <typename dist_t> struct BenchmarkFixture : public benchmark::Fixture {
  std::shared_ptr<Index<dist_t, int>> _index;
  std::string _dataset_filepath;
  uint32_t _ef_construction;
  uint32_t _ef_search;
  uint32_t _max_edges_per_node;
  uint32_t _dataset_size;
  uint32_t _dim;

  float *_data;
  cnpy::NpyArray _datafile;

  void SetUp(benchmark::State &state) override {
    _dataset_filepath = dataset_id_to_filepath[state.range(0)];

    _ef_construction = state.range(1);
    _ef_search = state.range(2);
    _max_edges_per_node = state.range(3);

    _dataset_size = state.range(4);
    _dim = state.range(5);

    // Load the dataset from the file
    _datafile = cnpy::npy_load(_dataset_filepath);
    if (_datafile.shape.size() != 2) {
      throw std::runtime_error("Invalid benchmark filename");
    }
    _data = _datafile.data<float>();

    // Construct the index with the given parameters
    auto distance = std::make_unique<dist_t>(_dim);
    _index = std::make_shared<Index<dist_t, int>>(
        /* dist = */ std::move(distance), /* dataset_size = */ _dataset_size,
        /* max_edges = */ _max_edges_per_node);
  }

  void TearDown(benchmark::State &state) override { _index.reset(); }
};

BENCHMARK_TEMPLATE_DEFINE_F(BenchmarkFixture, L2BasedBenchmarkRun,
                            SquaredL2Distance)
(benchmark::State &state) {
  // This will run the actual benchmark
  for (auto _ : state) {
    buildIndex<SquaredL2Distance>(
        this->_index, this->_data, this->_dataset_size,
        this->_max_edges_per_node, this->_dim, this->_ef_construction);

    // benchmark::DoNotOptimize(_index);
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(BenchmarkFixture, InnerProductBasedBenchmarkRun,
                            InnerProductDistance)
(benchmark::State &state) {
  // This will run the actual benchmark
  for (auto _ : state) {
    buildIndex<InnerProductDistance>(
        this->_index, this->_data, this->_dataset_size,
        this->_max_edges_per_node, this->_dim, this->_ef_construction);

    // benchmark::DoNotOptimize(_index);
  }

}

std::string validated_config_file() {
  char buffer[FILENAME_MAX];
  std::string config_file;

  if (GETCWD(buffer, FILENAME_MAX) != nullptr) {
    std::string current_working_dir(buffer);
    config_file = current_working_dir + "/benchmarks/config.yaml";
  } else {
    std::cerr << "Error: Could not get current working directory" << std::endl;
  }

  return config_file;
}

std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream token_stream(s);
  while (std::getline(token_stream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<std::string> getSpecifiedDatasets(int argc, char **argv) {
  std::vector<std::string> dataset_names;

  for (uint32_t i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if ((arg == "--datasets" || arg == "-d") && i + 1 < argc) {
      std::string datasets_str = argv[++i];
      dataset_names = split(datasets_str, ',');
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "\t--datasets, -d: comma-separated list of datasets to run"
                << std::endl;
      std::cout << "\t--help, -h: print this help message" << std::endl;
      exit(0);
    } else {
      std::cerr << "Error: Unknown option " << arg << std::endl;
      exit(1);
    }
  }

  return dataset_names;
}

int main(int argc, char **argv) {

  // Get the datasets to run from the command line
  auto specified_datasets = getSpecifiedDatasets(argc, argv);

  // Load benchmark config
  auto config_file = validated_config_file();
  BenchmarksConfig benchmarks_config =
      parseBenchmarkConfig(/* filename = */ config_file);

  uint32_t dataset_id = 0;
  for (const auto &dataset : benchmarks_config.datasets) {
    auto param_template =
        benchmarks_config
            .index_parameter_templates[dataset.index_parameter_template];

    if (std::find(specified_datasets.begin(), specified_datasets.end(),
                  dataset.name) == specified_datasets.end()) {
      continue;
    }

    dataset_id_to_filepath[dataset_id] = dataset.train_filepath;

    for (const auto &ef_construction : param_template.ef_construction) {
      for (const auto &ef_search : param_template.ef_search) {
        for (const auto &max_edges_per_node :
             param_template.max_edges_per_node) {

          if (dataset.metric_id == 0) {
            BENCHMARK_REGISTER_F(BenchmarkFixture, L2BasedBenchmarkRun)
                ->Args({static_cast<uint32_t>(dataset_id),
                        static_cast<uint32_t>(ef_construction),
                        static_cast<uint32_t>(ef_search),
                        static_cast<uint32_t>(max_edges_per_node),
                        static_cast<uint32_t>(dataset.train_size),
                        static_cast<uint32_t>(dataset.dim)});
          } else if (dataset.metric_id == 1) {
            BENCHMARK_REGISTER_F(BenchmarkFixture,
                                 InnerProductBasedBenchmarkRun)
                ->Args({static_cast<uint32_t>(dataset_id),
                        static_cast<uint32_t>(ef_construction),
                        static_cast<uint32_t>(ef_search),
                        static_cast<uint32_t>(max_edges_per_node),
                        static_cast<uint32_t>(dataset.train_size),
                        static_cast<uint32_t>(dataset.dim)});
          }
        }
      }
      dataset_id++;
    }
  }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
  }
