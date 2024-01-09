
#include "cnpy.h"
#include <flatnav/DistanceInterface.h>
#include <flatnav/Index.h>
#include <flatnav/distances/SquaredL2Distance.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>

using flatnav::DistanceInterface;
using flatnav::Index;
using flatnav::SquaredL2Distance;

int main(int argc, char **argv) {


  if (argc < 5) {
    std::cout << "Usage: ./build/load_mtx <train_data_path> <queries_path> "
                 "<groundtruth_path> <mtx_filename> "
              << std::endl;
    std::cout << "Example: ./build/load_mtx "
                 "<path-to-data>/data/"
                 "sift-128-euclidean/sift-128-euclidean.train.npy "
                 "path-to-data/data/"
                 "sift-128-euclidean/sift-128-euclidean.test.npy "
                 "path-to-data/data/"
                 "sift-128-euclidean/sift-128-euclidean.gtruth.npy "
                 "sift.mtx "
              << std::endl;
    return 1;
  }

  std::string train_data_path = argv[1];
  std::string queries_path = argv[2];
  std::string groundtruth_path = argv[3];
  std::string mtx_filename = argv[4];

  cnpy::NpyArray trainfile = cnpy::npy_load(train_data_path);
  cnpy::NpyArray queryfile = cnpy::npy_load(queries_path);
  cnpy::NpyArray truthfile = cnpy::npy_load(groundtruth_path);
  if ((queryfile.shape.size() != 2) || (truthfile.shape.size() != 2)) {
    return -1;
  }

  int dim = trainfile.shape[1];
  int dataset_size = trainfile.shape[0];

  float *data = trainfile.data<float>();
  float *queries = queryfile.data<float>();
  int *gtruth = truthfile.data<int>();
  int M = 32;

  std::cout << "constructing the index" << std::endl;
  auto distance = std::make_shared<flatnav::SquaredL2Distance>(dim);
  std::unique_ptr<flatnav::Index<flatnav::SquaredL2Distance, int>> index =
      std::make_unique<flatnav::Index<flatnav::SquaredL2Distance, int>>(
          distance, mtx_filename);

  std::vector<int> ef_searches{100, 200, 300, 500, 1000, 2000, 3000};
  int num_queries = queryfile.shape[0];
  int num_gtruth = truthfile.shape[1];
  int K = 100;

  std::cout << "Adding vectors to the index" << std::endl;
  for (int label = 0; label < dataset_size; label++) {
    float *element = data + (dim * label);
    uint32_t node_id;

    index->allocateNode(element, label, node_id);
  }

  std::cout << "Building graph links" << std::endl;
  index->buildGraphLinks();

  std::cout << "Querying" << std::endl;

  for (const auto &ef_search : ef_searches) {
    double mean_recall = 0;

    auto start_q = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_queries; i++) {
      float *q = queries + dim * i;
      int *g = gtruth + num_gtruth * i;

      std::vector<std::pair<float, int>> result =
          index->search(q, K, ef_search);

      double recall = 0;
      for (int j = 0; j < K; j++) {
        for (int l = 0; l < K; l++) {
          if (result[j].second == g[l]) {
            recall = recall + 1;
          }
        }
      }
      recall = recall / K;
      mean_recall = mean_recall + recall;
    }
    auto stop_q = std::chrono::high_resolution_clock::now();
    auto duration_q =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_q - start_q);
    std::cout << "[INFO] Mean Recall: " << mean_recall / num_queries
              << ", Duration: " << (float)(duration_q.count()) / num_queries
              << " for ef_search = " << ef_search << std::endl;
  }

  return 0;
}