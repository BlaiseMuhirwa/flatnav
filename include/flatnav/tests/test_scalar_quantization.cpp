
#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Index.h>
#include <flatnav/quantization/ScalarQuantizedDistance.h>
#include <cmath>
#include <random>
#include <vector>
#include "gtest/gtest.h"

namespace flatnav::testing {

using flatnav::distances::MetricType;
using flatnav::quantization::ScalarQuantizedDistance;

class ScalarQuantizationTest : public ::testing::Test {
 protected:
  static constexpr size_t k_dim = 128;
  static constexpr size_t k_num_vectors = 500;
  std::vector<float> data;

  void SetUp() override {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 10.0f);
    data.resize(k_num_vectors * k_dim);
    for (auto& v : data) {
      v = dist(rng);
    }
  }
};

TEST_F(ScalarQuantizationTest, TrainComputesCorrectMinMax) {
  ScalarQuantizedDistance<MetricType::L2> sq(k_dim);
  sq.train(data.data(), k_num_vectors);

  ASSERT_TRUE(sq.isTrained());

  // Verify min/max per dimension
  for (size_t d = 0; d < k_dim; d++) {
    float expected_min = data[d];
    float expected_max = data[d];
    for (size_t i = 1; i < k_num_vectors; i++) {
      float val = data[i * k_dim + d];
      if (val < expected_min) expected_min = val;
      if (val > expected_max) expected_max = val;
    }
    EXPECT_FLOAT_EQ(sq.minVals()[d], expected_min) << "dim=" << d;
    EXPECT_FLOAT_EQ(sq.maxVals()[d], expected_max) << "dim=" << d;
  }
}

TEST_F(ScalarQuantizationTest, QuantizeDequantizeRoundTrip) {
  ScalarQuantizedDistance<MetricType::L2> sq(k_dim);
  sq.train(data.data(), k_num_vectors);

  std::vector<int8_t> quantized(k_dim);
  std::vector<float> reconstructed(k_dim);

  for (size_t i = 0; i < k_num_vectors; i++) {
    const float* vec = data.data() + i * k_dim;
    sq.quantize(vec, quantized.data());
    sq.dequantize(quantized.data(), reconstructed.data());

    for (size_t d = 0; d < k_dim; d++) {
      float range = sq.maxVals()[d] - sq.minVals()[d];
      // Reconstruction error should be bounded by 1 quantization step
      float max_error = (range > 1e-10f) ? range / 255.0f : 0.0f;
      EXPECT_NEAR(reconstructed[d], vec[d], max_error + 1e-5f)
          << "vec=" << i << " dim=" << d;
    }
  }
}

TEST_F(ScalarQuantizationTest, SymmetricDistanceL2) {
  ScalarQuantizedDistance<MetricType::L2> sq(k_dim);
  sq.train(data.data(), k_num_vectors);

  std::vector<int8_t> q1(k_dim), q2(k_dim);
  sq.quantize(data.data(), q1.data());
  sq.quantize(data.data() + k_dim, q2.data());

  float dist = sq.distance(q1.data(), q2.data(), /* asymmetric = */ false);

  // Compute expected: squared L2 between quantized vectors
  float expected = 0;
  for (size_t d = 0; d < k_dim; d++) {
    float diff = static_cast<float>(q1[d]) - static_cast<float>(q2[d]);
    expected += diff * diff;
  }
  EXPECT_FLOAT_EQ(dist, expected);
}

TEST_F(ScalarQuantizationTest, AsymmetricDistanceL2) {
  ScalarQuantizedDistance<MetricType::L2> sq(k_dim);
  sq.train(data.data(), k_num_vectors);

  const float* query = data.data();
  std::vector<int8_t> stored(k_dim);
  sq.quantize(data.data() + k_dim, stored.data());

  float dist = sq.distance(query, stored.data(), /* asymmetric = */ true);

  // In asymmetric mode, the query gets quantized on-the-fly,
  // so the result should match quantized-vs-quantized
  std::vector<int8_t> query_quantized(k_dim);
  sq.quantize(query, query_quantized.data());

  float expected = 0;
  for (size_t d = 0; d < k_dim; d++) {
    float diff =
        static_cast<float>(query_quantized[d]) - static_cast<float>(stored[d]);
    expected += diff * diff;
  }
  EXPECT_FLOAT_EQ(dist, expected);
}

TEST_F(ScalarQuantizationTest, EndToEndL2) {
  constexpr size_t N = 200;
  constexpr int M = 32;
  constexpr int K = 10;
  constexpr int ef = 100;

  auto sq = ScalarQuantizedDistance<MetricType::L2>::create(k_dim);
  sq->train(data.data(), N);

  using IndexType = Index<ScalarQuantizedDistance<MetricType::L2>, int>;
  auto dist_ptr = std::unique_ptr<
      DistanceInterface<ScalarQuantizedDistance<MetricType::L2>>>(
      std::move(sq));

  IndexType index(std::move(dist_ptr), N, M, false, DataType::float32);

  // Add data
  std::vector<int> labels(N);
  std::iota(labels.begin(), labels.end(), 0);
  index.template addBatch<float>((void*)data.data(), labels, ef);

  // Search: each vector should find itself as nearest neighbor
  int self_found = 0;
  for (size_t i = 0; i < std::min(N, (size_t)50); i++) {
    const float* query = data.data() + i * k_dim;
    auto results = index.search(query, K, ef);
    // Check if the vector's own label appears in top-K
    for (auto& [dist, label] : results) {
      if (label == static_cast<int>(i)) {
        self_found++;
        break;
      }
    }
  }
  // With scalar quantization, we expect at least 80% self-recall
  EXPECT_GE(self_found, 40) << "Self-recall too low: " << self_found << "/50";
}

TEST_F(ScalarQuantizationTest, EndToEndIP) {
  constexpr size_t N = 200;
  constexpr int M = 32;
  constexpr int K = 10;
  constexpr int ef = 100;

  auto sq = ScalarQuantizedDistance<MetricType::IP>::create(k_dim);
  sq->train(data.data(), N);

  using IndexType = Index<ScalarQuantizedDistance<MetricType::IP>, int>;
  auto dist_ptr = std::unique_ptr<
      DistanceInterface<ScalarQuantizedDistance<MetricType::IP>>>(
      std::move(sq));

  IndexType index(std::move(dist_ptr), N, M, false, DataType::float32);

  std::vector<int> labels(N);
  std::iota(labels.begin(), labels.end(), 0);
  index.template addBatch<float>((void*)data.data(), labels, ef);

  // Just verify search doesn't crash and returns K results
  const float* query = data.data();
  auto results = index.search(query, K, ef);
  EXPECT_EQ(results.size(), K);
}

TEST_F(ScalarQuantizationTest, SerializationRoundTrip) {
  constexpr size_t N = 100;
  constexpr int M = 16;
  constexpr int K = 5;
  constexpr int ef = 64;

  auto sq = ScalarQuantizedDistance<MetricType::L2>::create(k_dim);
  sq->train(data.data(), N);

  using IndexType = Index<ScalarQuantizedDistance<MetricType::L2>, int>;
  auto dist_ptr = std::unique_ptr<
      DistanceInterface<ScalarQuantizedDistance<MetricType::L2>>>(
      std::move(sq));

  IndexType index(std::move(dist_ptr), N, M, false, DataType::float32);

  std::vector<int> labels(N);
  std::iota(labels.begin(), labels.end(), 0);
  index.template addBatch<float>((void*)data.data(), labels, ef);

  // Search before save
  const float* query = data.data();
  auto results_before = index.search(query, K, ef);

  // Save and reload
  std::string filename = "/tmp/test_sq_index.bin";
  index.saveIndex(filename);

  auto loaded_index = IndexType::loadIndex(filename);
  auto results_after = loaded_index->search(query, K, ef);

  // Results should be identical
  ASSERT_EQ(results_before.size(), results_after.size());
  for (size_t i = 0; i < results_before.size(); i++) {
    EXPECT_EQ(results_before[i].second, results_after[i].second)
        << "label mismatch at rank " << i;
    EXPECT_FLOAT_EQ(results_before[i].first, results_after[i].first)
        << "distance mismatch at rank " << i;
  }

  // Cleanup
  std::remove(filename.c_str());
}

TEST_F(ScalarQuantizationTest, SampledTrainingProducesValidMinMax) {
  // Train on all data first to get reference min/max
  ScalarQuantizedDistance<MetricType::L2> sq_full(k_dim);
  sq_full.train(data.data(), k_num_vectors);

  // Train on a sample (80% of the data — large enough to be close)
  size_t sample_size = 400;
  ScalarQuantizedDistance<MetricType::L2> sq_sampled(k_dim);
  sq_sampled.train(data.data(), k_num_vectors, sample_size);

  ASSERT_TRUE(sq_sampled.isTrained());

  for (size_t d = 0; d < k_dim; d++) {
    // Sampled min should be >= full min (it can't see values it didn't sample)
    EXPECT_GE(sq_sampled.minVals()[d], sq_full.minVals()[d]) << "dim=" << d;
    // Sampled max should be <= full max
    EXPECT_LE(sq_sampled.maxVals()[d], sq_full.maxVals()[d]) << "dim=" << d;

    // With 400/500 samples, the sampled min/max should be close to full
    float full_range = sq_full.maxVals()[d] - sq_full.minVals()[d];
    float tolerance = 0.3f * full_range;  // within 30% of the full range
    EXPECT_NEAR(sq_sampled.minVals()[d], sq_full.minVals()[d], tolerance)
        << "dim=" << d;
    EXPECT_NEAR(sq_sampled.maxVals()[d], sq_full.maxVals()[d], tolerance)
        << "dim=" << d;
  }
}

TEST_F(ScalarQuantizationTest, SampledTrainingWithLargeSampleUsesAll) {
  ScalarQuantizedDistance<MetricType::L2> sq_full(k_dim);
  sq_full.train(data.data(), k_num_vectors);

  // max_train_samples >= n should produce identical results to full training
  ScalarQuantizedDistance<MetricType::L2> sq_large(k_dim);
  sq_large.train(data.data(), k_num_vectors, k_num_vectors + 100);

  for (size_t d = 0; d < k_dim; d++) {
    EXPECT_FLOAT_EQ(sq_large.minVals()[d], sq_full.minVals()[d]) << "dim=" << d;
    EXPECT_FLOAT_EQ(sq_large.maxVals()[d], sq_full.maxVals()[d]) << "dim=" << d;
  }
}

}  // namespace flatnav::testing
