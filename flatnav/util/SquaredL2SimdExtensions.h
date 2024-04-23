#pragma once

#include <flatnav/util/SimdBaseTypes.h>

namespace flatnav::util {

#if defined(USE_AVX512)
static float computeL2_Avx512(const void *x, const void *y,
                              const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  // Align to 16-floats boundary
  const float *end_x = pointer_x + (dimension >> 4 << 4);
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

static float computeL2_Avx512_int8(const void *x, const void *y,
                                   const size_t &dimension) {
  int8_t *pointer_x = static_cast<int8_t *>(const_cast<void *>(x));
  int8_t *pointer_y = static_cast<int8_t *>(const_cast<void *>(y));

  // Align to 64-int8s boundary
  // // Mask the lower 6 bits to align down to the nearest multiple of 64
  const int8_t *end_x = pointer_x + (dimension & ~63);

  simd64int8 difference, v1, v2;

  __m512i sum = _mm512_setzero_si512();

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;

    // Squaring the difference and accumulating
    __m512i diff_int = difference.get();
    __m512i diff_squared = _mm512_mullo_epi16(diff_int, diff_int);

    // Convert squared differences from 16-bit to 32-bit before summing to avoid
    // overflow
    __m512i diff_squared_lo =
        _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(diff_squared, 0));
    __m512i diff_squared_hi =
        _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(diff_squared, 1));

    // Summing the squared differences
    sum = _mm512_add_epi32(sum, diff_squared_lo);
    sum = _mm512_add_epi32(sum, diff_squared_hi);
    pointer_x += 64;
    pointer_y += 64;
  }

  // Reduce the sum to a single float result
  int32_t sum_array[16];
  _mm512_storeu_si512(sum_array, sum);
  int32_t result = 0;
  for (int i = 0; i < 16; i++) {
    result += sum_array[i];
  }

  return static_cast<float>(result);
}

#endif // USE_AVX512

#if defined(USE_AVX)

static float computeL2_Avx2(const void *x, const void *y,
                            const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  const float *end_x = pointer_x + (dimension >> 4 << 4);
  simd8float32 difference, v1, v2;
  simd8float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 8;
    pointer_y += 8;
  }

  return sum.reduce_add();
}

#endif // USE_AVX

#if defined(USE_SSE)

static float computeL2_Sse(const void *x, const void *y,
                           const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  const float *end_x = pointer_x + (dimension >> 4 << 4);
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

// This function computes the L2 distance between two int8 vectors using SSE2
// instructions.
static float computeL2_Sse_int8(const void *x, const void *y,
                                const size_t &dimension) {
  int8_t *pointer_x = static_cast<int8_t *>(const_cast<void *>(x));
  int8_t *pointer_y = static_cast<int8_t *>(const_cast<void *>(y));

  __m128i sum = _mm_setzero_si128();
  size_t aligned_dimension = dimension & ~0xF;
  size_t i = 0;

  for (; i < aligned_dimension; i += 16) {
    __m128i vx = _mm_loadu_si128(reinterpret_cast<__m128i *>(pointer_x + i));
    __m128i vy = _mm_loadu_si128(reinterpret_cast<__m128i *>(pointer_y + i));
    __m128i diff = _mm_sub_epi8(vx, vy);

    // Convert to 16-bit and square
    __m128i diff_squared =
        _mm_madd_epi16(_mm_cvtepi8_epi16(diff), _mm_cvtepi8_epi16(diff));

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
  _mm_storeu_si128(reinterpret_cast<__m128i *>(buffer), sum);
  return static_cast<float>(buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                            partial_sum);
}

static float computeL2_Sse4Aligned(const void *x, const void *y,
                                   const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  const float *end_x = pointer_x + (dimension >> 2 << 2);
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

static float computeL2_SseWithResidual_16(const void *x, const void *y,
                                          const size_t &dimension) {

  size_t dimension_aligned = dimension >> 4 << 4;
  float aligned_distance = computeL2_Sse(x, y, dimension_aligned);
  size_t residual_dimension = dimension - dimension_aligned;
  float residual_distance = 0.0f;
  float *pointer_x =
      static_cast<float *>(const_cast<void *>(x)) + dimension_aligned;
  float *pointer_y =
      static_cast<float *>(const_cast<void *>(y)) + dimension_aligned;
  for (size_t i = 0; i < residual_dimension; i++) {
    float difference = *pointer_x - *pointer_y;
    residual_distance += difference * difference;
    pointer_x++;
    pointer_y++;
  }
  return aligned_distance + residual_distance;
}

static float computeL2_Sse4aligned(const void *x, const void *y,
                                   const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  const float *end_x = pointer_x + (dimension >> 2 << 2);
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

static float computeL2_SseWithResidual_4(const void *x, const void *y,
                                         const size_t &dimension) {
  size_t dimension_aligned = dimension >> 2 << 2;
  float aligned_distance = computeL2_Sse4aligned(x, y, dimension_aligned);
  size_t residual_dimension = dimension - dimension_aligned;
  float residual_distance = 0.0f;
  float *pointer_x =
      static_cast<float *>(const_cast<void *>(x)) + dimension_aligned;
  float *pointer_y =
      static_cast<float *>(const_cast<void *>(y)) + dimension_aligned;
  for (size_t i = 0; i < residual_dimension; i++) {
    float difference = *pointer_x - *pointer_y;
    residual_distance += difference * difference;
    pointer_x++;
    pointer_y++;
  }
  return aligned_distance + residual_distance;
}

#endif // USE_SSE

} // namespace flatnav::util