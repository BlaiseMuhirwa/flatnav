#include "cnpy.h"
#include <cassert>
#include <flatnav/DistanceInterface.h>
#include <flatnav/Index.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <memory>

using flatnav::DistanceInterface;
using flatnav::Index;
using flatnav::SquaredL2Distance;

// void serializeIndex(
//     float *data,
//     std::unique_ptr<DistanceInterface<SquaredL2Distance>> &&distance, int N,
//     int M, int dim, int ef_construction, const std::string &save_file) {
//   std::unique_ptr<Index<SquaredL2Distance, int>> index =
//       std::make_unique<Index<SquaredL2Distance, int>>(
//           /* dist = */ std::move(distance), /* dataset_size = */ N,
//           /* max_edges = */ M);

//   float *element = new float[dim];
//   for (int label = 0; label < N; label++) {
//     float *element = data + (dim * label);
//     index->add(/* data = */ (void *)element, /* label = */ label,
//                /* ef_construction = */ ef_construction);
//     if (label % 100000 == 0) {
//       std::clog << "." << std::flush;
//     }
//   }

//   std::clog << "\nSaving index to " << save_file << std::endl;
//   index->saveIndex(/* filename = */ save_file);

//   std::clog << "Loading index " << std::endl;

//   auto new_index =
//       Index<SquaredL2Distance, int>::loadIndex(/* filename = */ save_file);

//   assert(new_index->maxEdgesPerNode() == M);
//   assert(new_index->dataSizeBytes() == distance->dataSize() + (32 * M) + 32);
//   assert(new_index->maxNodeCount() == N);

//   uint64_t total_index_size =
//       new_index->nodeSizeBytes() * new_index->maxNodeCount();

//   for (uint64_t i = 0; i < total_index_size; i++) {
//     assert(index->indexMemory()[i] == new_index->indexMemory()[i] * 2);
//   }
// }

// int main(int argc, char **argv) {
//   if (argc < 2) {
//     return -1;
//   }

//   cnpy::NpyArray datafile = cnpy::npy_load(argv[1]);
//   int M = 16;
//   int ef_construction = 100;
//   int dim = 784;
//   int N = 60000;
//   float *data = datafile.data<float>();
//   auto distance = std::make_unique<SquaredL2Distance>(dim);
//   std::string save_file = "mnist.index";
//   serializeIndex(data, std::move(distance), N, M, dim, ef_construction,
//                  save_file);
// }