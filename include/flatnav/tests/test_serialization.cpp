#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Index.h>
#include <cassert>
#include <cstdio>  // for remove
#include <random>
#include "gtest/gtest.h"

using flatnav::Index;
using flatnav::distances::DistanceInterface;
using flatnav::distances::InnerProductDistance;
using flatnav::distances::SquaredL2Distance;
using flatnav::util::DataType;


namespace flatnav::testing {

static const uint32_t INDEXED_VECTORS = 10000;
static const uint32_t QUERY_VECTORS = 500;
static const uint32_t VEC_DIM = 500;
static const uint32_t K = 10;
static const uint32_t EF_SEARCH = 50;

// TODO: This is duplicated a couple times. Move it to a common testing
// utils file.
template<typename T>
std::vector<T> generateRandomVectors(uint32_t num_vectors, uint32_t dim) {
  std::vector<T> vectors(num_vectors * dim);
  for (uint32_t i = 0; i < num_vectors * dim; i++) {
    vectors[i] = static_cast<T>(rand()) / RAND_MAX;
  }
  return vectors;
}

template <typename T, typename dist_t, typename label_t>
void runTest(T* data, std::unique_ptr<DistanceInterface<dist_t>>&& distance, int N, int M, int dim,
             int ef_construction, const std::string& save_file) {
  auto data_size = distance->dataSize();
  auto data_type = distance->getDataType();

  std::unique_ptr<Index<dist_t, label_t>> index = std::make_unique<Index<dist_t, label_t>>(
      /* dist = */ std::move(distance), /* dataset_size = */ N,
      /* max_edges = */ M, /* collect_stats= */ false, /* data_type = */ data_type);

  std::vector<int> labels(N);
  std::iota(labels.begin(), labels.end(), 0);
  index->template addBatch<T>(data, labels, ef_construction);
  index->saveIndex(/* filename = */ save_file);

  auto new_index = Index<dist_t, label_t>::loadIndex(/* filename = */ save_file);

  ASSERT_EQ(new_index->maxEdgesPerNode(), M);
  ASSERT_EQ(new_index->dataSizeBytes(), index->dataSizeBytes());
  // 4 bytes for each node_id, 4 bytes for the label
  ASSERT_EQ(new_index->nodeSizeBytes(), data_size + (4 * M) + 4);
  ASSERT_EQ(new_index->maxNodeCount(), N);
  ASSERT_EQ(new_index->getDataType(), data_type);

  uint64_t total_index_size = new_index->nodeSizeBytes() * new_index->maxNodeCount();

  std::vector<T> queries = generateRandomVectors<T>(QUERY_VECTORS, dim);

  for (uint32_t i = 0; i < QUERY_VECTORS; i++) {
    T* q = queries.data() + (dim * i);

    std::vector<std::pair<float, int>> query_result = index->search(q, K, EF_SEARCH);

    std::vector<std::pair<float, int>> new_query_result = new_index->search(q, K, EF_SEARCH);

    for (uint32_t j = 0; j < K; j++) {
      ASSERT_EQ(query_result[j].first, new_query_result[j].first);
      ASSERT_EQ(query_result[j].second, new_query_result[j].second);
    }
  }
}

TEST(FlatnavSerializationTest, TestL2FloatIndexSerialization) {
  auto vectors = generateRandomVectors<float>(INDEXED_VECTORS, VEC_DIM);
  auto distance = std::make_unique<SquaredL2Distance<>>(VEC_DIM);
  std::string save_file = "l2_index.bin";

  uint32_t ef_construction = 100;
  uint32_t M = 16;

  runTest<float, SquaredL2Distance<DataType::float32>, int>(
      /* data = */ vectors.data(), /* distance */ std::move(distance),
      /* N = */ INDEXED_VECTORS, /* M = */ M, /* dim = */ VEC_DIM,
      /* ef_construction = */ ef_construction,
      /* save_file = */ save_file);

  EXPECT_EQ(std::remove(save_file.c_str()), 0);
}

TEST(FlatnavSerializationTest, TestL2Uint8IndexSerialization) {
  auto vectors = generateRandomVectors<uint8_t>(INDEXED_VECTORS, VEC_DIM);
  auto distance = std::make_unique<SquaredL2Distance<DataType::uint8>>(VEC_DIM);
  std::string save_file = "l2_index.bin";

  uint32_t ef_construction = 100;
  uint32_t M = 16;

  runTest<uint8_t, SquaredL2Distance<DataType::uint8>, int>(
      /* data = */ vectors.data(), /* distance */ std::move(distance),
      /* N = */ INDEXED_VECTORS, /* M = */ M, /* dim = */ VEC_DIM,
      /* ef_construction = */ ef_construction,
      /* save_file = */ save_file);

  EXPECT_EQ(std::remove(save_file.c_str()), 0);
}

TEST(FlatnavSerializationTest, TestL2Int8IndexSerialization) {
  auto vectors = generateRandomVectors<int8_t>(INDEXED_VECTORS, VEC_DIM);
  auto distance = std::make_unique<SquaredL2Distance<DataType::int8>>(VEC_DIM);
  std::string save_file = "l2_index.bin";

  uint32_t ef_construction = 100;
  uint32_t M = 16;

  runTest<int8_t, SquaredL2Distance<DataType::int8>, int>(
      /* data = */ vectors.data(), /* distance */ std::move(distance),
      /* N = */ INDEXED_VECTORS, /* M = */ M, /* dim = */ VEC_DIM,
      /* ef_construction = */ ef_construction,
      /* save_file = */ save_file);

  EXPECT_EQ(std::remove(save_file.c_str()), 0);
}


TEST(FlatnavSerializationTest, TestInnerProductFloatIndexSerialization) {
  auto vectors = generateRandomVectors<float>(INDEXED_VECTORS, VEC_DIM);
  auto distance = std::make_unique<InnerProductDistance<>>(VEC_DIM);
  std::string save_file = "ip_index.bin";

  uint32_t ef_construction = 100;
  uint32_t M = 16;

  runTest<float, InnerProductDistance<DataType::float32>, int>(
      /* data = */ vectors.data(), /* distance = */ std::move(distance),
      /* N = */ INDEXED_VECTORS, /* M = */ M, /* dim = */ VEC_DIM,
      /* ef_construction */ ef_construction, /* save_file = */ save_file);

  EXPECT_EQ(std::remove(save_file.c_str()), 0);
}

TEST(FlatnavSerializationTest, TestInnerProductUint8IndexSerialization) {
  auto vectors = generateRandomVectors<uint8_t>(INDEXED_VECTORS, VEC_DIM);
  auto distance = std::make_unique<InnerProductDistance<DataType::uint8>>(VEC_DIM);
  std::string save_file = "ip_index.bin";

  uint32_t ef_construction = 100;
  uint32_t M = 16;

  runTest<uint8_t, InnerProductDistance<DataType::uint8>, int>(
      /* data = */ vectors.data(), /* distance = */ std::move(distance),
      /* N = */ INDEXED_VECTORS, /* M = */ M, /* dim = */ VEC_DIM,
      /* ef_construction */ ef_construction, /* save_file = */ save_file);

  EXPECT_EQ(std::remove(save_file.c_str()), 0);
}

TEST(FlatnavSerializationTest, TestInnerProductInt8IndexSerialization) {
  auto vectors = generateRandomVectors<int8_t>(INDEXED_VECTORS, VEC_DIM);
  auto distance = std::make_unique<InnerProductDistance<DataType::int8>>(VEC_DIM);
  std::string save_file = "ip_index.bin";

  uint32_t ef_construction = 100;
  uint32_t M = 16;

  runTest<int8_t, InnerProductDistance<DataType::int8>, int>(
      /* data = */ vectors.data(), /* distance = */ std::move(distance),
      /* N = */ INDEXED_VECTORS, /* M = */ M, /* dim = */ VEC_DIM,
      /* ef_construction */ ef_construction, /* save_file = */ save_file);

  EXPECT_EQ(std::remove(save_file.c_str()), 0);
}

}  // namespace flatnav::testing