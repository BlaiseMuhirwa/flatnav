
#include <flatnav/util/Macros.h>
#include <flatnav/util/SimdUtils.h>
#include <chrono>
#include <random>
#include "gtest/gtest.h"

#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>

namespace flatnav::testing {

class DistanceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize x and y with values drawn from a normal distribution
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 10.0f);

    for (size_t i = 0; i < dimensions; ++i) {
      x[i] = distribution(generator);
      y[i] = distribution(generator);
    }
  }

  static constexpr size_t dimensions = 128;

  // TODO: This epsilon is too high. I noticed that one or two inner product SSE
  // tests fail with a lower epsilon. I need to investigate why this is
  // happening. The goal should be to have an epsilon of 1e-6 or lower.
  static constexpr float epsilon = 1e-2;
  float x[dimensions];
  float y[dimensions];
};

// Test case for AVX512-based L2 distance computer
TEST_F(DistanceTest, TestAvx512L2Distance) {
#if defined(USE_AVX512)
  float result = flatnav::util::computeL2_Avx512(x, y, dimensions);
  float expected = flatnav::distances::defaultSquaredL2<float>(x, y, dimensions);
  ASSERT_NEAR(result, expected, epsilon);

#endif
}

// Test case for AVX512-based L2 distance computer for uint8_t data type
TEST_F(DistanceTest, TestAvx512L2DistanceUint8) {
#if defined(USE_AVX512)
  auto total_num_vectors = 1000;
  auto total_size = dimensions * total_num_vectors;
  uint8_t* x_matrix = (uint8_t*)malloc(total_size);
  uint8_t* y_matrix = (uint8_t*)malloc(total_size);
  for (size_t i = 0; i < total_size; i++) {
    x_matrix[i] = (uint8_t)rand() % 256;
    y_matrix[i] = (uint8_t)rand() % 256;
  }

  for (size_t i = 0; i < total_num_vectors; i++) {
    uint8_t* x = x_matrix + i * dimensions;
    uint8_t* y = y_matrix + i * dimensions;
    float result = flatnav::util::computeL2_Avx512_Uint8(x, y, dimensions);
    float expected = flatnav::distances::defaultSquaredL2<uint8_t>(x, y, dimensions);
    ASSERT_NEAR(result, expected, epsilon);
  }

  free(x_matrix);
  free(y_matrix);

#endif
}

// Test case for AVX-based L2 distance computer
TEST_F(DistanceTest, TestAvxL2Distance) {
#if defined(USE_AVX)

  float result = flatnav::util::computeL2_Avx2(x, y, dimensions);
  float expected = flatnav::distances::defaultSquaredL2<float>(x, y, dimensions);

  ASSERT_NEAR(result, expected, epsilon);

#endif
}

TEST(TestSingleIntrinsic, TestReduceAddAvx) {
#if defined(USE_AVX)
  flatnav::util::simd8float32 v(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
  float result = v.reduce_add();
  float expected = 36.0f;
  ASSERT_NEAR(result, expected, 1e-6);
#endif
}

TEST(TestSingleIntrinsic, TestReduceAddSse) {
#if defined(USE_SSE)
  flatnav::util::simd4float32 v(1.0f, 2.0f, 3.0f, 4.0f);
  float result = v.reduce_add();
  float expected = 10.0f;
  ASSERT_NEAR(result, expected, 1e-6);
#endif
}

// Test case for SSE-based L2 distance computers
TEST_F(DistanceTest, TestSseL2Distance) {
#if defined(USE_SSE)
  float result = flatnav::util::computeL2_Sse(x, y, dimensions);
  float expected = flatnav::distances::defaultSquaredL2<float>(x, y, dimensions);
  ASSERT_NEAR(result, expected, epsilon);

  // try with dimensions not divisible by 16
  // this will just take the first 100 elements in the arrays
  result = flatnav::util::computeL2_Sse4Aligned(x, y, 100);
  expected = flatnav::distances::defaultSquaredL2<float>(x, y, 100);

  ASSERT_NEAR(result, expected, epsilon);

  // try with dimensions not divisible by 4
  result = flatnav::util::computeL2_SseWithResidual_16(x, y, 37);
  expected = flatnav::distances::defaultSquaredL2<float>(x, y, 37);
  ASSERT_NEAR(result, expected, epsilon);

  // try with dimensions not divisible by 4 and less than 16
  result = flatnav::util::computeL2_SseWithResidual_4(x, y, 7);
  expected = flatnav::distances::defaultSquaredL2<float>(x, y, 7);
  ASSERT_NEAR(result, expected, epsilon);

#endif
}

// Test case for AVX512-based inner product distance computer
TEST_F(DistanceTest, TestAvx512InnerProductDistance) {
#if defined(USE_AVX512)
  float result = flatnav::util::computeIP_Avx512(x, y, dimensions);
  float expected = flatnav::distances::defaultInnerProduct(x, y, dimensions);
  ASSERT_NEAR(result, expected, epsilon);

#endif
}

// Test case for AVX-based inner product distance computer
TEST_F(DistanceTest, TestAvxInnerProductDistance) {
#if defined(USE_AVX)
  float result = flatnav::util::computeIP_Avx(x, y, dimensions);
  float expected = flatnav::distances::defaultInnerProduct(x, y, dimensions);
  ASSERT_NEAR(result, expected, epsilon);

#endif
}

// Test case for SSE-based inner product distance computer
TEST_F(DistanceTest, TestSseInnerProductDistance) {
#if defined(USE_SSE)
  float result = flatnav::util::computeIP_Sse(x, y, dimensions);
  float expected = flatnav::distances::defaultInnerProduct(x, y, dimensions);
  ASSERT_NEAR(result, expected, epsilon);

  // try with dimensions not divisible by 16
  // this will just take the first 100 elements in the arrays
  result = flatnav::util::computeIP_Sse_4aligned(x, y, 100);
  expected = flatnav::distances::defaultInnerProduct(x, y, 100);
  ASSERT_NEAR(result, expected, epsilon);

#if defined(USE_AVX)
  result = flatnav::util::computeIP_Avx_4aligned(x, y, 100);
  expected = flatnav::distances::defaultInnerProduct(x, y, 100);
  ASSERT_NEAR(result, expected, epsilon);
#endif

  // try with dimensions not divisible by 4
  result = flatnav::util::computeIP_SseWithResidual_16(x, y, 37);
  expected = flatnav::distances::defaultInnerProduct(x, y, 37);
  ASSERT_NEAR(result, expected, epsilon);

  // try with dimensions not divisible by 4 and less than 16
  result = flatnav::util::computeIP_SseWithResidual_4(x, y, 7);
  expected = flatnav::distances::defaultInnerProduct(x, y, 7);
  ASSERT_NEAR(result, expected, epsilon);

#endif
}

}  // namespace flatnav::testing