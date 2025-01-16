#pragma once

#include <flatnav/util/SimdUtils.h>

namespace flatnav::util {

#if defined(USE_AVX512)
static float computeL2_Avx512(const void* x, const void* y, const size_t& dimension) {
  float* pointer_x = static_cast<float*>(const_cast<void*>(x));
  float* pointer_y = static_cast<float*>(const_cast<void*>(y));

  // Align to 16-floats boundary
  const float* end_x = pointer_x + (dimension >> 4 << 4);
  simd16float32 difference, v1, v2;

  simd16float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 16;
    pointer_y += 16;
  }
  return sum.reduce_add();
}

/**
 * @todo Make this support dimensions that are not multiples of 64
 */
static float computeL2_Avx512_Uint8(const void* x, const void* y, const size_t& dimension) {
  const uint8_t* pointer_x = static_cast<const uint8_t*>(x);
  const uint8_t* pointer_y = static_cast<const uint8_t*>(y);

  // Initialize sum to zero
  __m512i sum = _mm512_setzero_si512();

  // Loop over the input arrays
  for (size_t i = 0; i < dimension; i += 64) {
    // Load 64 bytes from each array
    __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(pointer_x + i));
    __m512i v2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(pointer_y + i));

    // Unpack to 16-bit integers to avoid overflow
    __m512i v1_lo = _mm512_unpacklo_epi8(v1, _mm512_setzero_si512());
    __m512i v1_hi = _mm512_unpackhi_epi8(v1, _mm512_setzero_si512());
    __m512i v2_lo = _mm512_unpacklo_epi8(v2, _mm512_setzero_si512());
    __m512i v2_hi = _mm512_unpackhi_epi8(v2, _mm512_setzero_si512());

    // Compute differences
    __m512i diff_lo = _mm512_sub_epi16(v1_lo, v2_lo);
    __m512i diff_hi = _mm512_sub_epi16(v1_hi, v2_hi);

    // Square the differences
    __m512i diff_squared_lo = _mm512_madd_epi16(diff_lo, diff_lo);
    __m512i diff_squared_hi = _mm512_madd_epi16(diff_hi, diff_hi);

    // Accumulate the results
    sum = _mm512_add_epi32(sum, diff_squared_lo);
    sum = _mm512_add_epi32(sum, diff_squared_hi);
  }

  // Sum all elements in the sum vector
  __m256i sum256 = _mm512_extracti64x4_epi64(sum, 0);
  sum256 = _mm256_add_epi32(sum256, _mm512_extracti64x4_epi64(sum, 1));
  sum256 = _mm256_hadd_epi32(sum256, sum256);
  sum256 = _mm256_hadd_epi32(sum256, sum256);

  int32_t buffer[8];
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer), sum256);

  int32_t total_sum = buffer[0] + buffer[4];

  return static_cast<float>(total_sum);
}

#endif  // USE_AVX512

#if defined(USE_AVX)

static float computeL2_Avx2(const void* x, const void* y, const size_t& dimension) {
  float* pointer_x = static_cast<float*>(const_cast<void*>(x));
  float* pointer_y = static_cast<float*>(const_cast<void*>(y));

  const float* end_x = pointer_x + (dimension & ~7);
  simd8float32 difference, v1, v2;
  simd8float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;

    pointer_x += 8;
    pointer_y += 8;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;

    pointer_x += 8;
    pointer_y += 8;
  }

  float result[8];
  sum.storeu(result);
  return result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];
}

#endif  // USE_AVX

#if defined(USE_SSE)

static float computeL2_Sse(const void* x, const void* y, const size_t& dimension) {
  float* pointer_x = static_cast<float*>(const_cast<void*>(x));
  float* pointer_y = static_cast<float*>(const_cast<void*>(y));

  const float* end_x = pointer_x + (dimension >> 4 << 4);
  simd4float32 difference, v1, v2;
  simd4float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 4;
    pointer_y += 4;
  }

  return sum.reduce_add();
}

#if defined(USE_SSE4_1)

// This function computes the L2 distance between two int8 vectors using SSE2
// instructions.
static float computeL2_Sse_int8(const void* x, const void* y, const size_t& dimension) {
  int8_t* pointer_x = static_cast<int8_t*>(const_cast<void*>(x));
  int8_t* pointer_y = static_cast<int8_t*>(const_cast<void*>(y));

  __m128i sum = _mm_setzero_si128();
  size_t aligned_dimension = dimension & ~0xF;
  size_t i = 0;

  for (; i < aligned_dimension; i += 16) {
    __m128i vx = _mm_loadu_si128(reinterpret_cast<__m128i*>(pointer_x + i));
    __m128i vy = _mm_loadu_si128(reinterpret_cast<__m128i*>(pointer_y + i));
    __m128i diff = _mm_sub_epi8(vx, vy);

    // Convert to 16-bit and square
    __m128i diff_squared = _mm_madd_epi16(_mm_cvtepi8_epi16(diff), _mm_cvtepi8_epi16(diff));

    // Accumulate in 32-bit integer
    sum = _mm_add_epi32(sum, diff_squared);
  }

  // Handle the remaining elements
  int32_t partial_sum = 0;
  for (; i < dimension; i++) {
    int diff = pointer_x[i] - pointer_y[i];
    partial_sum += diff * diff;
  }

  // Reduce sum
  int32_t buffer[4];
  _mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), sum);
  return static_cast<float>(buffer[0] + buffer[1] + buffer[2] + buffer[3] + partial_sum);
}

#endif  // USE_SSE4_1

static float computeL2_Sse4Aligned(const void* x, const void* y, const size_t& dimension) {
  float* pointer_x = static_cast<float*>(const_cast<void*>(x));
  float* pointer_y = static_cast<float*>(const_cast<void*>(y));

  const float* end_x = pointer_x + (dimension >> 2 << 2);
  simd4float32 difference, v1, v2;
  simd4float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 4;
    pointer_y += 4;
  }

  return sum.reduce_add();
}

static float computeL2_SseWithResidual_16(const void* x, const void* y, const size_t& dimension) {

  size_t dimension_aligned = dimension >> 4 << 4;
  float aligned_distance = computeL2_Sse(x, y, dimension_aligned);
  size_t residual_dimension = dimension - dimension_aligned;
  float residual_distance = 0.0f;
  float* pointer_x = static_cast<float*>(const_cast<void*>(x)) + dimension_aligned;
  float* pointer_y = static_cast<float*>(const_cast<void*>(y)) + dimension_aligned;
  for (size_t i = 0; i < residual_dimension; i++) {
    float difference = *pointer_x - *pointer_y;
    residual_distance += difference * difference;
    pointer_x++;
    pointer_y++;
  }
  return aligned_distance + residual_distance;
}

static float computeL2_Sse4aligned(const void* x, const void* y, const size_t& dimension) {
  float* pointer_x = static_cast<float*>(const_cast<void*>(x));
  float* pointer_y = static_cast<float*>(const_cast<void*>(y));

  const float* end_x = pointer_x + (dimension >> 2 << 2);
  simd4float32 difference, v1, v2;
  simd4float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 4;
    pointer_y += 4;
  }

  return sum.reduce_add();
}

static float computeL2_SseWithResidual_4(const void* x, const void* y, const size_t& dimension) {
  size_t dimension_aligned = dimension >> 2 << 2;
  float aligned_distance = computeL2_Sse4aligned(x, y, dimension_aligned);
  size_t residual_dimension = dimension - dimension_aligned;
  float residual_distance = 0.0f;
  float* pointer_x = static_cast<float*>(const_cast<void*>(x)) + dimension_aligned;
  float* pointer_y = static_cast<float*>(const_cast<void*>(y)) + dimension_aligned;
  for (size_t i = 0; i < residual_dimension; i++) {
    float difference = *pointer_x - *pointer_y;
    residual_distance += difference * difference;
    pointer_x++;
    pointer_y++;
  }
  return aligned_distance + residual_distance;
}

#endif  // USE_SSE

}  // namespace flatnav::util