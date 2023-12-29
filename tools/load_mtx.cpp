
#include "cnpy.h"
#include <flatnav/DistanceInterface.h>
#include <flatnav/Index.h>
#include <flatnav/distances/SquaredL2Distance.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>


using flatnav::Index;
using flatnav::SquaredL2Distance;
using flatnav::DistanceInterface;

struct Graph {
  std::vector<std::vector<uint32_t>> adjacency_list;
  int num_vertices;
};

// Function to load a graph from a Matrix Market file
Graph loadGraphFromMatrixMarket(const char *filename) {
  std::ifstream input_file;
  input_file.open(filename);

  if (!input_file.is_open()) {
    std::cerr << "Error opening file" << std::endl;
    exit(1);
  }

  std::string line;
  // Skip the header
  while (std::getline(input_file, line)) {
    if (line[0] != '%')
      break;
  }

  std::istringstream iss(line);
  int num_vertices, numEdges;
  iss >> num_vertices >> num_vertices >> numEdges;

  // Initialize graph
  Graph graph;
  graph.num_vertices = num_vertices;
  graph.adjacency_list.resize(num_vertices);

  int u, v;
  while (input_file >> u >> v) {
    // Adjust for 1-based indexing in Matrix Market format
    u--;
    v--;
    graph.adjacency_list[u].push_back(v);
  }

  input_file.close();
  return graph;
}

void writeToMatrixMarketFile(
    std::vector<std::vector<uint32_t>> &outdegree_table, std::string &filename,
    uint32_t num_edges) {
  std::ofstream output_file;
  output_file.open(filename);

  if (!output_file.is_open()) {
    std::cerr << "Error opening file" << std::endl;
    exit(1);
  }

  output_file << "%%MatrixMarket matrix coordinate integer general"
              << std::endl;
  output_file << outdegree_table.size() << " " << outdegree_table.size() << " "
              << num_edges << std::endl;

  for (int i = 0; i < outdegree_table.size(); i++) {
    for (int j = 0; j < outdegree_table[i].size(); j++) {
      output_file << i + 1 << " " << outdegree_table[i][j] + 1 << std::endl;
    }
  }

  output_file.close();
}

int main() {
  // Replace with your filename
  const char *ground_truth_file =
      "/Users/blaisemunyampirwa/Desktop/flatnav-experimental/data/"
      "deep-image-96-angular/deep-image-96-angular.gtruth.npy";
  const char *train_file =
      "/Users/blaisemunyampirwa/Desktop/flatnav-experimental/data/"
      "deep-image-96-angular/deep-image-96-angular.train.npy";
  const char *queries_file =
      "/Users/blaisemunyampirwa/Desktop/flatnav-experimental/data/"
      "deep-image-96-angular/deep-image-96-angular.test.npy";
  const char *sift_mtx =
      "/Users/blaisemunyampirwa/Desktop/flatnav-experimental/data/"
      "deep-image-96-angular/deep96.mtx";

  Graph g = loadGraphFromMatrixMarket(sift_mtx);

  cnpy::NpyArray trainfile = cnpy::npy_load(train_file);
  cnpy::NpyArray queryfile = cnpy::npy_load(queries_file);
  cnpy::NpyArray truthfile = cnpy::npy_load(ground_truth_file);
  if ((queryfile.shape.size() != 2) || (truthfile.shape.size() != 2)) {
    return -1;
  }

  float *data = trainfile.data<float>();
  float *queries = queryfile.data<float>();
  int *gtruth = truthfile.data<int>();
  int dim = 96;
  int M = 32;
  int dataset_size = 9990000;

  std::cout << "constructing the index" << std::endl;
  auto distance = std::make_shared<flatnav::SquaredL2Distance>(dim);
  std::unique_ptr<flatnav::Index<flatnav::SquaredL2Distance, int>> index =
      std::make_unique<flatnav::Index<flatnav::SquaredL2Distance, int>>(
          distance, g.adjacency_list);

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
