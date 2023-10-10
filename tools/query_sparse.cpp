#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "cnpy.h"
#include <algorithm>
#include <flatnav/Index.h>
#include <string>

typedef std::vector<std::vector<uint32_t>> graph_t;

int main(int argc, char **argv) {

  if (argc < 9) {
    std::clog << "Usage: " << std::endl;
    std::clog << "query <space> <index> <queries> <gtruth> <ef_search> <k> "
                 "<Reorder ID> <graph.mtx>"
              << std::endl;
    std::clog << "\t <data> <queries> <gtruth>: .npy files (float, float, int) "
                 "from ann-benchmarks"
              << std::endl;
    std::clog << "\t <M>: int number of links" << std::endl;
    std::clog << "\t <ef_construction>: int " << std::endl;
    std::clog << "\t <ef_search>: int,int,int,int...,int " << std::endl;
    std::clog << "\t <k>: number of neighbors " << std::endl;
    return -1;
  }

  int space_ID = std::stoi(argv[1]);
  std::string indexfilename(argv[2]);

  std::vector<int> ef_searches;
  std::stringstream ss(argv[5]);
  int element = 0;
  while (ss >> element) {
    ef_searches.push_back(element);
    if (ss.peek() == ',')
      ss.ignore();
  }
  int k = std::stoi(argv[6]);
  int reorder_ID = std::stoi(argv[7]);

  cnpy::NpyArray queryfile = cnpy::npy_load(argv[3]);
  cnpy::NpyArray truthfile = cnpy::npy_load(argv[4]);
  if ((queryfile.shape.size() != 2) || (truthfile.shape.size() != 2)) {
    return -1;
  }

  int Nq = queryfile.shape[0];
  int dim = queryfile.shape[1];
  int n_gt = truthfile.shape[1];
  if (k > n_gt) {
    std::cerr
        << "K is larger than the number of precomputed ground truth neighbors"
        << std::endl;
    return -1;
  }

  std::clog << "Loading " << Nq << " queries" << std::endl;
  float *queries = queryfile.data<float>();
  std::clog << "Loading " << Nq << " ground truth results with k = " << k
            << std::endl;
  int *gtruth = truthfile.data<int>();

  SpaceInterface<float> *space;
  if (space_ID == 0) {
    space = new L2Space(dim);
  } else {
    space = new InnerProductSpace(dim);
  }

  Index<float, int> index(space, indexfilename);

  std::cout << "Flashing..." << std::endl;
  auto start_f = std::chrono::high_resolution_clock::now();
  std::ifstream graphstream(argv[8]);
  graphstream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  graphstream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  graph_t graph(index.size());
  uint32_t ID;
  uint32_t edge;
  while (graphstream >> ID) {
    graphstream >> edge;
    graph[ID - 1].push_back(edge - 1);
  }
  index.flash(graph);

  auto stop_f = std::chrono::high_resolution_clock::now();
  auto duration_f =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop_f - start_f);
  std::cout << "Flash time: " << duration_f.count() / 1000.0 << " seconds."
            << std::endl;

  if (reorder_ID == 1) {
    std::clog << "Using GORDER" << std::endl;
    std::clog << "Reordering: " << std::endl;
    auto start_r = std::chrono::high_resolution_clock::now();
    index.reorder(Index<float, int>::GraphOrder::GORDER);
    auto stop_r = std::chrono::high_resolution_clock::now();
    auto duration_r =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_r - start_r);
    std::clog << "Reorder time: " << (float)(duration_r.count()) / (1000.0)
              << " seconds" << std::endl;
  } else if (reorder_ID == 2) {
    std::clog << "Using IN-DEG-SORT" << std::endl;
    std::clog << "Reordering: " << std::endl;
    auto start_r = std::chrono::high_resolution_clock::now();
    index.reorder(Index<float, int>::GraphOrder::IN_DEG);
    auto stop_r = std::chrono::high_resolution_clock::now();
    auto duration_r =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_r - start_r);
    std::clog << "Reorder time: " << (float)(duration_r.count()) / (1000.0)
              << " seconds" << std::endl;
  } else if (reorder_ID == 3) {
    std::clog << "Using OUT-DEG-SORT" << std::endl;
    std::clog << "Reordering: " << std::endl;
    auto start_r = std::chrono::high_resolution_clock::now();
    index.reorder(Index<float, int>::GraphOrder::OUT_DEG);
    auto stop_r = std::chrono::high_resolution_clock::now();
    auto duration_r =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_r - start_r);
    std::clog << "Reorder time: " << (float)(duration_r.count()) / (1000.0)
              << " seconds" << std::endl;
  } else if (reorder_ID == 4) {
    std::clog << "Using Reverse-Cuthill-McKee" << std::endl;
    std::clog << "Reordering: " << std::endl;
    auto start_r = std::chrono::high_resolution_clock::now();
    index.reorder(Index<float, int>::GraphOrder::RCM);
    auto stop_r = std::chrono::high_resolution_clock::now();
    auto duration_r =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_r - start_r);
    std::clog << "Reorder time: " << (float)(duration_r.count()) / (1000.0)
              << " seconds" << std::endl;
  } else if (reorder_ID == 5) {
    std::clog << "Using HUBSORT" << std::endl;
    std::clog << "Reordering: " << std::endl;
    auto start_r = std::chrono::high_resolution_clock::now();
    index.reorder(Index<float, int>::GraphOrder::HUB_SORT);
    auto stop_r = std::chrono::high_resolution_clock::now();
    auto duration_r =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_r - start_r);
    std::clog << "Reorder time: " << (float)(duration_r.count()) / (1000.0)
              << " seconds" << std::endl;
  } else if (reorder_ID == 6) {
    std::clog << "Using HUBCLUSTER" << std::endl;
    std::clog << "Reordering: " << std::endl;
    auto start_r = std::chrono::high_resolution_clock::now();
    index.reorder(Index<float, int>::GraphOrder::HUB_CLUSTER);
    auto stop_r = std::chrono::high_resolution_clock::now();
    auto duration_r =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_r - start_r);
    std::clog << "Reorder time: " << (float)(duration_r.count()) / (1000.0)
              << " seconds" << std::endl;
  } else if (reorder_ID == 7) {
    std::clog << "Using DBG" << std::endl;
    std::clog << "Reordering: " << std::endl;
    auto start_r = std::chrono::high_resolution_clock::now();
    index.reorder(Index<float, int>::GraphOrder::DBG);
    auto stop_r = std::chrono::high_resolution_clock::now();
    auto duration_r =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_r - start_r);
    std::clog << "Reorder time: " << (float)(duration_r.count()) / (1000.0)
              << " seconds" << std::endl;
  } else {
    std::clog << "No reordering" << std::endl;
  }

  for (int &ef_search : ef_searches) {
    double mean_recall = 0;

    auto start_q = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < Nq; i++) {
      float *q = queries + dim * i;
      int *g = gtruth + n_gt * i;

      std::vector<std::pair<float, int>> result = index.search(q, k, ef_search);

      double recall = 0;
      for (int j = 0; j < k; j++) {
        for (int l = 0; l < k; l++) {
          if (result[j].second == g[l]) {
            recall = recall + 1;
          }
        }
      }
      recall = recall / k;
      mean_recall = mean_recall + recall;
    }
    auto stop_q = std::chrono::high_resolution_clock::now();
    auto duration_q =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_q - start_q);
    std::cout << mean_recall / Nq << "," << (float)(duration_q.count()) / Nq
              << std::endl;
  }

  return 0;
}