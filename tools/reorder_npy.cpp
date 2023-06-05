#include "../flatnav/Index.h"
#include "../flatnav/distances/InnerProductDistance.h"
#include "../flatnav/distances/SquaredL2Distance.h"
#include "cnpy.h"
#include <algorithm>
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

int main(int argc, char **argv) {

  if (argc < 4) {
    std::clog << "Usage: " << std::endl;
    std::clog << "reorder_npy <index_in> <reorder_id> <index_out> ";
    std::clog << "[ <queries> <space> <num_queries> <ef_search> ]";
    std::clog << "Positional arguments:" << std::endl;
    std::clog << "\t <index_in>: Filename for input index (float32 index)."
              << std::endl;
    std::clog
        << "\t <reorder_id>: Which reordering algorithm to use? 0:none "
           "1:gorder 2:indegsort 3:outdegsort 4:RCM 5:hubsort 6:hubcluster "
           "7:DBG 8:corder 91:profiled_gorder 94:profiled_rcm 41:RCM+gorder"
        << std::endl;
    std::clog << "\t <index_out>: Filename for output index (float32 index)."
              << std::endl;
    std::clog << "Profiling arguments:" << std::endl;
    std::clog << "\t <queries>: Filename for profiling queries (NPY file)."
              << std::endl;
    std::clog << "\t <space>: Integer distance ID: 0 for L2 distance, 1 for "
                 "inner product (angular distance)."
              << std::endl;
    std::clog << "\t <n_queries>: Number of queries to use for profiling."
              << std::endl;
    std::clog
        << "\t <ef_search>: Candidate list size for search with profiling."
        << std::endl;
    return -1;
  }

  std::string infile(argv[1]);
  std::string outfile(argv[3]);

  int reorder_ID = std::stoi(argv[2]);
  bool is_profiling = ((reorder_ID >= 90) && (reorder_ID < 100));
  if (is_profiling) {
    // We are doing profiling, so we need the extra arguments.
    if (argc < 8) {
      std::clog << "Incorrect arguments for profiled reordering. Run with no "
                   "arguments for help."
                << std::endl;
      return -1;
    }
    cnpy::NpyArray queryfile = cnpy::npy_load(argv[4]);
    int num_queries_check = queryfile.shape[0];
    int dim = queryfile.shape[1];
    int num_queries = std::stoi(argv[6]);
    int ef_search = std::stoi(argv[7]);

    std::clog << "Reading " << num_queries << " queries of "
              << num_queries_check << " total queries of dimension " << dim
              << "." << std::endl;
    if (num_queries_check != num_queries) {
      std::clog << "Warning: Using only " << num_queries << " points of total "
                << num_queries_check << "." << std::endl;
    }
    float *queries = queryfile.data<float>();
    std::clog << "Read " << num_queries << " queries for profiling."
              << std::endl;

    int metric_id = std::stoi(argv[5]);
    if (metric_id == 0) {
      auto distance = std::make_unique<SquaredL2Distance>(dim);

    } else if (metric_id == 1) {
      auto distance = std::make_unique<InnerProductDistance>(dim);
    } else {
      throw std::invalid_argument("Provided metric ID " +
                                  std::to_string(metric_id) + "is invalid.");
    }
    auto index = new Index<SquaredL2Distance, int>(/* stream */ infile);

    // SpaceInterface<float>* space;
    // if (space_ID == 0){
    //     space = new L2Space(dim);
    // } else {
    //     space = new InnerProductSpace(dim);
    // }
    // Index<float, int> index(space, infile);

    if (reorder_ID == 91) {
      std::clog << "Using profile-based GORDER" << std::endl;
      std::clog << "Reordering" << std::endl;
      auto start_r = std::chrono::high_resolution_clock::now();
      index.profile_reorder(queries, num_queries, ef_search,
                            Index<float, int>::ProfileOrder::GORDER);
      auto stop_r = std::chrono::high_resolution_clock::now();
      auto duration_r = std::chrono::duration_cast<std::chrono::milliseconds>(
          stop_r - start_r);
      std::clog << "Reorder time: " << (float)(duration_r.count()) / (1000.0)
                << " seconds" << std::endl;
    } else if (reorder_ID == 94) {
      std::clog << "Using profile-based RCM" << std::endl;
      std::clog << "Reordering" << std::endl;
      auto start_r = std::chrono::high_resolution_clock::now();
      index.profile_reorder(queries, num_queries, ef_search,
                            Index<float, int>::ProfileOrder::RCM);
      auto stop_r = std::chrono::high_resolution_clock::now();
      auto duration_r = std::chrono::duration_cast<std::chrono::milliseconds>(
          stop_r - start_r);
      std::clog << "Reorder time: " << (float)(duration_r.count()) / (1000.0)
                << " seconds" << std::endl;
    } else {
      std::clog << "No reordering" << std::endl;
    }
    std::clog << "Saving index." << std::endl;
    index.save(outfile);
  } else {
    // we can do reordering without knowing anything about the space or
    // dimensions.
    L2Space space(0);
    Index<float, int> index(&space, infile);

    if (reorder_ID == 1) {
      std::clog << "Using GORDER" << std::endl;
      std::clog << "Reordering: " << std::endl;
      auto start_r = std::chrono::high_resolution_clock::now();
      index.reorder(Index<float, int>::GraphOrder::GORDER);
      auto stop_r = std::chrono::high_resolution_clock::now();
      auto duration_r = std::chrono::duration_cast<std::chrono::milliseconds>(
          stop_r - start_r);
      std::clog << "Reorder time: " << (float)(duration_r.count()) / (1000.0)
                << " seconds" << std::endl;
    } else {
      std::clog << "No reordering" << std::endl;
    }
    std::clog << "Saving index." << std::endl;
    index.save(outfile);
  }

  return 0;
}
