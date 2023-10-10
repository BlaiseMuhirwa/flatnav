#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "cnpy.h"
#include <algorithm>
#include <flatnav/DistanceInterface.h>
#include <flatnav/Index.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <string>

using flatnav::DistanceInterface;
using flatnav::Index;
using flatnav::InnerProductDistance;
using flatnav::SquaredL2Distance;

typedef std::vector<std::vector<uint32_t>> graph_t;

template <typename dist_t>
void writeGraphToMatrixMarket(const std::string &index_filename,
                              const std::string &mtx_filename) {
  auto index = Index<dist_t, int>::loadIndex(index_filename);
  graph_t graph = index->graph();

  int num_nodes = graph.size();
  int num_edges = 0;

  for (auto &edges : graph) {
    num_edges += edges.size();
  }

  std::ofstream mtx_file(mtx_filename);
  if (!mtx_file.is_open()) {
    std::cerr << "Could not open file: " << mtx_filename << std::endl;
    exit(-1);
  }

  // %%MatrixMarket matrix coordinate pattern general
  // N N M
  // N = # nodes, M = # edges
  mtx_file << "\%\%MatrixMarket matrix coordinate pattern general" << std::endl;
  mtx_file << num_nodes << " " << num_nodes << " " << num_edges << std::endl;

  for (int node = 0; node < num_nodes; node++) {
    for (auto &edge : graph[node]) {
      mtx_file << node + 1 << " " << edge + 1 << std::endl;
    }
  }

  mtx_file.close();
}

int main(int argc, char **argv) {

  if (argc < 4) {
    std::clog << "Usage: " << std::endl;
    std::clog << "graphstats <index>" << std::endl;
    std::clog << "\t <index>: .index file" << std::endl;
    std::clog << "\t <metric_id>: 0 = L2, 1 = Inner Product" << std::endl;
    std::clog << "\t <graph.mtx>: output file" << std::endl;
    return -1;
  }

  std::string index_filename(argv[1]);
  uint32_t metric_id = std::stoi(argv[2]);
  std::string mtx_filename(argv[3]);
  flatnav::METRIC_TYPE metric_type = metric_id == 0
                                         ? flatnav::METRIC_TYPE::EUCLIDEAN
                                         : flatnav::METRIC_TYPE::INNER_PRODUCT;

  if (metric_type == flatnav::METRIC_TYPE::EUCLIDEAN) {
    writeGraphToMatrixMarket<SquaredL2Distance>(index_filename = index_filename,
                                                mtx_filename = mtx_filename);

  } else if (metric_type == flatnav::METRIC_TYPE::INNER_PRODUCT) {
    writeGraphToMatrixMarket<InnerProductDistance>(
        index_filename = index_filename, mtx_filename = mtx_filename);
  }
}