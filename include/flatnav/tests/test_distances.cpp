
#include <flatnav/util/Macros.h>
#include <flatnav/util/SimdUtils.h>
#include <chrono>
#include <random>
#include <vector>
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
  float result = flatnav::util::compute_l2_avx512(x, y, dimensions);
  float expected = flatnav::distances::default_squared_l2<float>(x, y, dimensions);
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
    float result = flatnav::util::compute_l2_avx512_uint8(x, y, dimensions);
    float expected = flatnav::distances::default_squared_l2<uint8_t>(x, y, dimensions);
    ASSERT_NEAR(result, expected, epsilon);
  }

  free(x_matrix);
  free(y_matrix);

#endif
}

// Test case for AVX-based L2 distance computer
TEST_F(DistanceTest, TestAvxL2Distance) {
#if defined(USE_AVX)

  float result = flatnav::util::compute_l2_avx2(x, y, dimensions);
  float expected = flatnav::distances::default_squared_l2<float>(x, y, dimensions);

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
  float result = flatnav::util::compute_l2_sse(x, y, dimensions);
  float expected = flatnav::distances::default_squared_l2<float>(x, y, dimensions);
  ASSERT_NEAR(result, expected, epsilon);

  // try with dimensions not divisible by 16
  // this will just take the first 100 elements in the arrays
  result = flatnav::util::compute_l2_sse_4aligned(x, y, 100);
  expected = flatnav::distances::default_squared_l2<float>(x, y, 100);

  ASSERT_NEAR(result, expected, epsilon);

  // try with dimensions not divisible by 4
  result = flatnav::util::compute_l2_sse_residual_16(x, y, 37);
  expected = flatnav::distances::default_squared_l2<float>(x, y, 37);
  ASSERT_NEAR(result, expected, epsilon);

  // try with dimensions not divisible by 4 and less than 16
  result = flatnav::util::compute_l2_sse_residual_4(x, y, 7);
  expected = flatnav::distances::default_squared_l2<float>(x, y, 7);
  ASSERT_NEAR(result, expected, epsilon);

#endif
}

// Test case for AVX512-based inner product distance computer
TEST_F(DistanceTest, TestAvx512InnerProductDistance) {
#if defined(USE_AVX512)
  float result = flatnav::util::compute_ip_avx512(x, y, dimensions);
  float expected = flatnav::distances::default_inner_product(x, y, dimensions);
  ASSERT_NEAR(result, expected, epsilon);

#endif
}

// Test case for AVX-based inner product distance computer
TEST_F(DistanceTest, TestAvxInnerProductDistance) {
#if defined(USE_AVX)
  float result = flatnav::util::compute_ip_avx(x, y, dimensions);
  float expected = flatnav::distances::default_inner_product(x, y, dimensions);
  ASSERT_NEAR(result, expected, epsilon);

#endif
}

// Test case for SSE-based inner product distance computer
TEST_F(DistanceTest, TestSseInnerProductDistance) {
#if defined(USE_SSE)
  float result = flatnav::util::compute_ip_sse(x, y, dimensions);
  float expected = flatnav::distances::default_inner_product(x, y, dimensions);
  ASSERT_NEAR(result, expected, epsilon);

  // try with dimensions not divisible by 16
  // this will just take the first 100 elements in the arrays
  result = flatnav::util::compute_ip_sse_4aligned(x, y, 100);
  expected = flatnav::distances::default_inner_product(x, y, 100);
  ASSERT_NEAR(result, expected, epsilon);

#if defined(USE_AVX)
  result = flatnav::util::compute_ip_avx_4aligned(x, y, 100);
  expected = flatnav::distances::default_inner_product(x, y, 100);
  ASSERT_NEAR(result, expected, epsilon);
#endif

  // try with dimensions not divisible by 4
  result = flatnav::util::compute_ip_sse_residual_16(x, y, 37);
  expected = flatnav::distances::default_inner_product(x, y, 37);
  ASSERT_NEAR(result, expected, epsilon);

  // try with dimensions not divisible by 4 and less than 16
  result = flatnav::util::compute_ip_sse_residual_4(x, y, 7);
  expected = flatnav::distances::default_inner_product(x, y, 7);
  ASSERT_NEAR(result, expected, epsilon);

#endif
}

// ---------------------------------------------------------------------------
// int8 SIMD distance tests
//
// Each test generates 1000 random int8 vector pairs in the full [-128, 127]
// range and compares the SIMD kernel result against the scalar fallback.
// We test two dimensions per kernel:
//   - 128: aligned to all SIMD widths (multiple of 16, 32, 64)
//   - 100: NOT a multiple of 16/32/64, exercises the scalar tail loop
//
// Since all arithmetic is integer (accumulated in int32, then cast to float),
// results should be exact. We use epsilon = 1.0f as a safety margin.
// ---------------------------------------------------------------------------

static constexpr size_t kInt8NumVectors = 1000;
static constexpr float kInt8Epsilon = 1.0f;

// Helper: generate random int8 test data with a fixed seed for reproducibility.
static void generateInt8TestData(int8_t* x, int8_t* y, size_t total_size, unsigned seed = 42) {
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(-128, 127);
  for (size_t i = 0; i < total_size; i++) {
    x[i] = static_cast<int8_t>(distribution(generator));
    y[i] = static_cast<int8_t>(distribution(generator));
  }
}

// Helper: run a SIMD kernel against the scalar baseline for multiple vector
// pairs at two different dimensions (aligned and residual).
template <typename KernelFn, typename BaselineFn>
static void runInt8KernelTest(KernelFn kernel, BaselineFn baseline,
                               size_t aligned_dim, size_t residual_dim) {
  // --- Test with aligned dimension ---
  {
    size_t total_size = aligned_dim * kInt8NumVectors;
    std::vector<int8_t> x_data(total_size), y_data(total_size);
    generateInt8TestData(x_data.data(), y_data.data(), total_size, 42);

    for (size_t i = 0; i < kInt8NumVectors; i++) {
      const int8_t* xp = x_data.data() + i * aligned_dim;
      const int8_t* yp = y_data.data() + i * aligned_dim;
      float result = kernel(xp, yp, aligned_dim);
      float expected = baseline(xp, yp, aligned_dim);
      ASSERT_NEAR(result, expected, kInt8Epsilon)
          << "Mismatch at vector " << i << " with aligned dimension " << aligned_dim;
    }
  }

  // --- Test with residual dimension (exercises scalar tail loop) ---
  {
    size_t total_size = residual_dim * kInt8NumVectors;
    std::vector<int8_t> x_data(total_size), y_data(total_size);
    generateInt8TestData(x_data.data(), y_data.data(), total_size, 123);

    for (size_t i = 0; i < kInt8NumVectors; i++) {
      const int8_t* xp = x_data.data() + i * residual_dim;
      const int8_t* yp = y_data.data() + i * residual_dim;
      float result = kernel(xp, yp, residual_dim);
      float expected = baseline(xp, yp, residual_dim);
      ASSERT_NEAR(result, expected, kInt8Epsilon)
          << "Mismatch at vector " << i << " with residual dimension " << residual_dim;
    }
  }
}

// --- L2 int8 kernel tests ---

TEST(Int8DistanceTest, TestSseL2Int8) {
#if defined(USE_SSE4_1)
  runInt8KernelTest(
      flatnav::util::compute_l2_sse_int8,
      flatnav::distances::default_squared_l2<int8_t>,
      128,  // aligned: multiple of 16
      100   // residual: not a multiple of 16
  );
#endif
}

TEST(Int8DistanceTest, TestAvx2L2Int8) {
#if defined(USE_AVX)
  runInt8KernelTest(
      flatnav::util::compute_l2_avx2_int8,
      flatnav::distances::default_squared_l2<int8_t>,
      128,  // aligned: multiple of 32
      100   // residual: not a multiple of 32
  );
#endif
}

TEST(Int8DistanceTest, TestAvx512L2Int8) {
#if defined(USE_AVX512)
  runInt8KernelTest(
      flatnav::util::compute_l2_avx512_int8,
      flatnav::distances::default_squared_l2<int8_t>,
      128,  // aligned: multiple of 32 (AVX512 kernel processes 32 bytes/iter)
      100   // residual
  );
#endif
}

// --- IP int8 kernel tests ---

TEST(Int8DistanceTest, TestSseIPInt8) {
#if defined(USE_SSE4_1)
  runInt8KernelTest(
      flatnav::util::compute_ip_sse_int8,
      flatnav::distances::default_inner_product<int8_t>,
      128,  // aligned: multiple of 16
      100   // residual
  );
#endif
}

TEST(Int8DistanceTest, TestAvx2IPInt8) {
#if defined(USE_AVX)
  runInt8KernelTest(
      flatnav::util::compute_ip_avx2_int8,
      flatnav::distances::default_inner_product<int8_t>,
      128,  // aligned: multiple of 32
      100   // residual
  );
#endif
}

TEST(Int8DistanceTest, TestAvx512IPInt8) {
#if defined(USE_AVX512)
  runInt8KernelTest(
      flatnav::util::compute_ip_avx512_int8,
      flatnav::distances::default_inner_product<int8_t>,
      128,  // aligned: multiple of 32
      100   // residual
  );
#endif
}

// ---------------------------------------------------------------------------
// uint8 SIMD distance tests
//
// Same structure as int8 tests, but with uint8 data in the full [0, 255]
// range. The key difference in the SIMD kernels is zero-extension (cvtepu8)
// instead of sign-extension (cvtepi8).
// ---------------------------------------------------------------------------

static constexpr size_t kUint8NumVectors = 1000;
static constexpr float kUint8Epsilon = 1.0f;

// Helper: generate random uint8 test data with a fixed seed for reproducibility.
static void generateUint8TestData(uint8_t* x, uint8_t* y, size_t total_size, unsigned seed = 42) {
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(0, 255);
  for (size_t i = 0; i < total_size; i++) {
    x[i] = static_cast<uint8_t>(distribution(generator));
    y[i] = static_cast<uint8_t>(distribution(generator));
  }
}

// Helper: run a SIMD kernel against the scalar baseline for multiple uint8
// vector pairs at two different dimensions (aligned and residual).
template <typename KernelFn, typename BaselineFn>
static void runUint8KernelTest(KernelFn kernel, BaselineFn baseline,
                                size_t aligned_dim, size_t residual_dim) {
  // --- Test with aligned dimension ---
  {
    size_t total_size = aligned_dim * kUint8NumVectors;
    std::vector<uint8_t> x_data(total_size), y_data(total_size);
    generateUint8TestData(x_data.data(), y_data.data(), total_size, 42);

    for (size_t i = 0; i < kUint8NumVectors; i++) {
      const uint8_t* xp = x_data.data() + i * aligned_dim;
      const uint8_t* yp = y_data.data() + i * aligned_dim;
      float result = kernel(xp, yp, aligned_dim);
      float expected = baseline(xp, yp, aligned_dim);
      ASSERT_NEAR(result, expected, kUint8Epsilon)
          << "Mismatch at vector " << i << " with aligned dimension " << aligned_dim;
    }
  }

  // --- Test with residual dimension (exercises scalar tail loop) ---
  {
    size_t total_size = residual_dim * kUint8NumVectors;
    std::vector<uint8_t> x_data(total_size), y_data(total_size);
    generateUint8TestData(x_data.data(), y_data.data(), total_size, 123);

    for (size_t i = 0; i < kUint8NumVectors; i++) {
      const uint8_t* xp = x_data.data() + i * residual_dim;
      const uint8_t* yp = y_data.data() + i * residual_dim;
      float result = kernel(xp, yp, residual_dim);
      float expected = baseline(xp, yp, residual_dim);
      ASSERT_NEAR(result, expected, kUint8Epsilon)
          << "Mismatch at vector " << i << " with residual dimension " << residual_dim;
    }
  }
}

// --- L2 uint8 kernel tests ---

TEST(Uint8DistanceTest, TestSseL2Uint8) {
#if defined(USE_SSE4_1)
  runUint8KernelTest(
      flatnav::util::compute_l2_sse_uint8,
      flatnav::distances::default_squared_l2<uint8_t>,
      128,  // aligned: multiple of 16
      100   // residual: not a multiple of 16
  );
#endif
}

TEST(Uint8DistanceTest, TestAvx2L2Uint8) {
#if defined(USE_AVX)
  runUint8KernelTest(
      flatnav::util::compute_l2_avx2_uint8,
      flatnav::distances::default_squared_l2<uint8_t>,
      128,  // aligned: multiple of 32
      100   // residual: not a multiple of 32
  );
#endif
}

TEST(Uint8DistanceTest, TestAvx512L2Uint8) {
#if defined(USE_AVX512)
  runUint8KernelTest(
      flatnav::util::compute_l2_avx512_uint8,
      flatnav::distances::default_squared_l2<uint8_t>,
      128,  // aligned: multiple of 32 (AVX512 kernel processes 32 bytes/iter)
      100   // residual
  );
#endif
}

// --- IP uint8 kernel tests ---

TEST(Uint8DistanceTest, TestSseIPUint8) {
#if defined(USE_SSE4_1)
  runUint8KernelTest(
      flatnav::util::compute_ip_sse_uint8,
      flatnav::distances::default_inner_product<uint8_t>,
      128,  // aligned: multiple of 16
      100   // residual
  );
#endif
}

TEST(Uint8DistanceTest, TestAvx2IPUint8) {
#if defined(USE_AVX)
  runUint8KernelTest(
      flatnav::util::compute_ip_avx2_uint8,
      flatnav::distances::default_inner_product<uint8_t>,
      128,  // aligned: multiple of 32
      100   // residual
  );
#endif
}

TEST(Uint8DistanceTest, TestAvx512IPUint8) {
#if defined(USE_AVX512)
  runUint8KernelTest(
      flatnav::util::compute_ip_avx512_uint8,
      flatnav::distances::default_inner_product<uint8_t>,
      128,  // aligned: multiple of 32
      100   // residual
  );
#endif
}

// --- Dispatcher end-to-end tests ---
// These test whichever SIMD path the build machine actually supports,
// ensuring the dispatch logic routes correctly.

TEST(Uint8DistanceTest, TestL2DispatcherUint8) {
  constexpr size_t dim = 128;
  constexpr size_t dim_residual = 100;
  std::vector<uint8_t> x_data(dim), y_data(dim);
  generateUint8TestData(x_data.data(), y_data.data(), dim, 77);

  float result = flatnav::distances::SquaredL2Impl<uint8_t>::computeDistance(
      x_data.data(), y_data.data(), dim);
  float expected = flatnav::distances::default_squared_l2<uint8_t>(
      x_data.data(), y_data.data(), dim);
  ASSERT_NEAR(result, expected, kUint8Epsilon);

  // Also test with a residual dimension
  std::vector<uint8_t> x_res(dim_residual), y_res(dim_residual);
  generateUint8TestData(x_res.data(), y_res.data(), dim_residual, 88);

  result = flatnav::distances::SquaredL2Impl<uint8_t>::computeDistance(
      x_res.data(), y_res.data(), dim_residual);
  expected = flatnav::distances::default_squared_l2<uint8_t>(
      x_res.data(), y_res.data(), dim_residual);
  ASSERT_NEAR(result, expected, kUint8Epsilon);
}

TEST(Uint8DistanceTest, TestIPDispatcherUint8) {
  constexpr size_t dim = 128;
  constexpr size_t dim_residual = 100;
  std::vector<uint8_t> x_data(dim), y_data(dim);
  generateUint8TestData(x_data.data(), y_data.data(), dim, 77);

  float result = flatnav::distances::InnerProductImpl<uint8_t>::computeDistance(
      x_data.data(), y_data.data(), dim);
  float expected = flatnav::distances::default_inner_product<uint8_t>(
      x_data.data(), y_data.data(), dim);
  ASSERT_NEAR(result, expected, kUint8Epsilon);

  // Also test with a residual dimension
  std::vector<uint8_t> x_res(dim_residual), y_res(dim_residual);
  generateUint8TestData(x_res.data(), y_res.data(), dim_residual, 88);

  result = flatnav::distances::InnerProductImpl<uint8_t>::computeDistance(
      x_res.data(), y_res.data(), dim_residual);
  expected = flatnav::distances::default_inner_product<uint8_t>(
      x_res.data(), y_res.data(), dim_residual);
  ASSERT_NEAR(result, expected, kUint8Epsilon);
}

TEST(Int8DistanceTest, TestL2DispatcherInt8) {
  constexpr size_t dim = 128;
  constexpr size_t dim_residual = 100;
  std::vector<int8_t> x_data(dim), y_data(dim);
  generateInt8TestData(x_data.data(), y_data.data(), dim, 77);

  float result = flatnav::distances::SquaredL2Impl<int8_t>::computeDistance(
      x_data.data(), y_data.data(), dim);
  float expected = flatnav::distances::default_squared_l2<int8_t>(
      x_data.data(), y_data.data(), dim);
  ASSERT_NEAR(result, expected, kInt8Epsilon);

  // Also test with a residual dimension
  std::vector<int8_t> x_res(dim_residual), y_res(dim_residual);
  generateInt8TestData(x_res.data(), y_res.data(), dim_residual, 88);

  result = flatnav::distances::SquaredL2Impl<int8_t>::computeDistance(
      x_res.data(), y_res.data(), dim_residual);
  expected = flatnav::distances::default_squared_l2<int8_t>(
      x_res.data(), y_res.data(), dim_residual);
  ASSERT_NEAR(result, expected, kInt8Epsilon);
}

TEST(Int8DistanceTest, TestIPDispatcherInt8) {
  constexpr size_t dim = 128;
  constexpr size_t dim_residual = 100;
  std::vector<int8_t> x_data(dim), y_data(dim);
  generateInt8TestData(x_data.data(), y_data.data(), dim, 77);

  float result = flatnav::distances::InnerProductImpl<int8_t>::computeDistance(
      x_data.data(), y_data.data(), dim);
  float expected = flatnav::distances::default_inner_product<int8_t>(
      x_data.data(), y_data.data(), dim);
  ASSERT_NEAR(result, expected, kInt8Epsilon);

  // Also test with a residual dimension
  std::vector<int8_t> x_res(dim_residual), y_res(dim_residual);
  generateInt8TestData(x_res.data(), y_res.data(), dim_residual, 88);

  result = flatnav::distances::InnerProductImpl<int8_t>::computeDistance(
      x_res.data(), y_res.data(), dim_residual);
  expected = flatnav::distances::default_inner_product<int8_t>(
      x_res.data(), y_res.data(), dim_residual);
  ASSERT_NEAR(result, expected, kInt8Epsilon);
}

}  // namespace flatnav::testing
