#pragma once

#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/util/SIMDIntrinsics.h>

#include <cstddef> // for size_t

namespace flatnav {

// The rest of the file contains specialized distance function implementations.
// These are heavily inspired by the ones from hnswlib, but with some
// refactoring. They are optimized for speed rather than readability.
//
// The naming convention lists the distance, dimension / chunk size,
// exact / residual dimension size, and instruction set. For example,
// L2SqrSIMD16ExtAVX512 is the L2-squared distance, using SIMD to process
// chunks of 16 dimensions at once, for problems where the dimension is an
// exact multiple of 16, using AVX512 instructions.

static float L2Sqr(const void *x, const void *y, size_t &dimension) {
  float *p_x = (float *)x;
  float *p_y = (float *)y;

  float result = 0;
  for (size_t i = 0; i < dimension; i++) {
    float diff = *p_x - *p_y;
    p_x++;
    p_y++;
    result += diff * diff;
  }
  return result;
}

#if defined(USE_AVX512)
static float L2SqrSIMD16ExtAVX512(const void *x, const void *y,
                                  size_t &dimension) {

  float *p_x = (float *)x;
  float *p_y = (float *)y;
  float PORTABLE_ALIGN64 tmp_result[16];
  size_t num_size_16_chunks = dimension >> 4;

  const float *p_x_end = p_x + (num_size_16_chunks << 4);

  __m512 diff, v1, v2;
  __m512 sum = _mm512_set1_ps(0);

  while (p_x < p_x_end) {
    v1 = _mm512_loadu_ps(p_x);
    p_x += 16;
    v2 = _mm512_loadu_ps(p_y);
    p_y += 16;
    diff = _mm512_sub_ps(v1, v2);
    sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
  }

  _mm512_store_ps(tmp_result, sum);
  float res = tmp_result[0] + tmp_result[1] + tmp_result[2] + tmp_result[3] +
              tmp_result[4] + tmp_result[5] + tmp_result[6] + tmp_result[7] +
              tmp_result[8] + tmp_result[9] + tmp_result[10] + tmp_result[11] +
              tmp_result[12] + tmp_result[13] + tmp_result[14] + tmp_result[15];
  return res;
}

static float L2SqrSIMD16ResAVX512(const void *x, const void *y,
                                  size_t &dimension) {

  size_t num_chunk_dims = dimension >> 4 << 4;
  float result = L2SqrSIMD16ExtAVX512(x, y, dimension);
  float *p_x = (float *)x + num_chunk_dims;
  float *p_y = (float *)y + num_chunk_dims;

  size_t num_leftover_dims = dimension - num_chunk_dims;
  float result_tail = L2Sqr(p_x, p_y, num_leftover_dims);
  return result + result_tail;
}
#endif

#if defined(USE_AVX)
// Favor using AVX if available.
static float L2SqrSIMD16ExtAVX(const void *x, const void *y,
                               size_t &dimension) {

  float *p_x = (float *)x;
  float *p_y = (float *)y;
  float PORTABLE_ALIGN32 tmp_result[8];
  size_t num_size_16_chunks = dimension >> 4;

  const float *p_x_end = p_x + (num_size_16_chunks << 4);

  __m256 diff, v1, v2;
  __m256 sum = _mm256_set1_ps(0);

  while (p_x < p_x_end) {
    v1 = _mm256_loadu_ps(p_x);
    p_x += 8;
    v2 = _mm256_loadu_ps(p_y);
    p_y += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

    v1 = _mm256_loadu_ps(p_x);
    p_x += 8;
    v2 = _mm256_loadu_ps(p_y);
    p_y += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
  }

  _mm256_store_ps(tmp_result, sum);
  return tmp_result[0] + tmp_result[1] + tmp_result[2] + tmp_result[3] +
         tmp_result[4] + tmp_result[5] + tmp_result[6] + tmp_result[7];
}

static float L2SqrSIMD16ResAVX(const void *x, const void *y,
                               size_t &dimension) {

  size_t num_chunk_dims = dimension >> 4 << 4;
  float result = L2SqrSIMD16ExtAVX(x, y, dimension);
  float *p_x = (float *)x + num_chunk_dims;
  float *p_y = (float *)y + num_chunk_dims;

  size_t num_leftover_dims = dimension - num_chunk_dims;
  float result_tail = L2Sqr(p_x, p_y, num_leftover_dims);
  return result + result_tail;
}
#endif

#if defined(USE_SSE)
static float L2SqrSIMD16ExtSSE(const void *x, const void *y,
                               size_t &dimension) {

  float *p_x = (float *)x;
  float *p_y = (float *)y;
  float PORTABLE_ALIGN32 tmp_result[8];
  size_t num_size_16_chunks = dimension >> 4;

  const float *p_x_end = p_x + (num_size_16_chunks << 4);

  __m128 diff, v1, v2;
  __m128 sum = _mm_set1_ps(0);

  while (p_x < p_x_end) {
    //_mm_prefetch((char*)(p_y + 16), _MM_HINT_T0);
    v1 = _mm_loadu_ps(p_x);
    p_x += 4;
    v2 = _mm_loadu_ps(p_y);
    p_y += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

    v1 = _mm_loadu_ps(p_x);
    p_x += 4;
    v2 = _mm_loadu_ps(p_y);
    p_y += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

    v1 = _mm_loadu_ps(p_x);
    p_x += 4;
    v2 = _mm_loadu_ps(p_y);
    p_y += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

    v1 = _mm_loadu_ps(p_x);
    p_x += 4;
    v2 = _mm_loadu_ps(p_y);
    p_y += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
  }

  _mm_store_ps(tmp_result, sum);
  return tmp_result[0] + tmp_result[1] + tmp_result[2] + tmp_result[3];
}

static float L2SqrSIMD16ResSSE(const void *x, const void *y,
                               size_t &dimension) {

  size_t num_chunk_dims = dimension >> 4 << 4;
  float result = L2SqrSIMD16ExtSSE(x, y, dimension);
  float *p_x = (float *)x + num_chunk_dims;
  float *p_y = (float *)y + num_chunk_dims;

  size_t num_leftover_dims = dimension - num_chunk_dims;
  float result_tail = L2Sqr(p_x, p_y, num_leftover_dims);
  return result + result_tail;
}

static float L2SqrSIMD4ExtSSE(const void *x, const void *y, size_t &dimension) {

  float PORTABLE_ALIGN32 tmp_result[8];
  float *p_x = (float *)x;
  float *p_y = (float *)y;

  size_t num_size_4_chunks = dimension >> 2;

  const float *p_x_end = p_x + (num_size_4_chunks << 2);

  __m128 diff, v1, v2;
  __m128 sum = _mm_set1_ps(0);

  while (p_x < p_x_end) {
    v1 = _mm_loadu_ps(p_x);
    p_x += 4;
    v2 = _mm_loadu_ps(p_y);
    p_y += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
  }
  _mm_store_ps(tmp_result, sum);
  return tmp_result[0] + tmp_result[1] + tmp_result[2] + tmp_result[3];
}

static float L2SqrSIMD4ResSSE(const void *x, const void *y, size_t &dimension) {

  // size_t dimension = *((size_t *) qty_ptr);
  size_t num_chunk_dims = dimension >> 2 << 2;

  float result = L2SqrSIMD4ExtSSE(x, y, num_chunk_dims);
  size_t num_leftover_dims = dimension - num_chunk_dims;

  float *p_x = (float *)x + num_chunk_dims;
  float *p_y = (float *)y + num_chunk_dims;
  float result_tail = L2Sqr(p_x, p_y, num_leftover_dims);

  return result + result_tail;
}
#endif

// TODO: Check whether the use of the "exact dimension match" distances
// is actually any faster than the general sized ones.

#if defined(USE_AVX512)
class SquaredL2Distance_SIMD16AVX512 : public SquaredL2Distance {
  // Specialization for processing size-16 chunks with AVX512.
  float distanceImpl(const void *x, const void *y) {
    auto dimension = this->getDimension();
    return L2SqrSIMD16ResAVX512(x, y, dimension);
  }
}
#endif

#if defined(USE_AVX)
class SquaredL2Distance_SIMD16AVX : public SquaredL2Distance {
  // Specialization for processing size-16 chunks with AVX.
  float distanceImpl(const void *x, const void *y) {
    auto dimension = this->getDimension();
    return L2SqrSIMD16ResAVX(x, y, dimension);
  }
}
#endif

#if defined(USE_SSE)
class SquaredL2Distance_SIMD16SSE : public SquaredL2Distance {
  // Specialization for processing size-16 chunks with SSE.
  float distanceImpl(const void *x, const void *y) {
    auto dimension = this->getDimension();
    return L2SqrSIMD16ResSSE(x, y, dimension);
  }
};

class SquaredL2Distance_SIMD4SSE : public SquaredL2Distance {
  // Specialization for processing size-4 chunks with SSE.
  float distanceImpl(const void *x, const void *y) {
    auto dimension = this->getDimension();
    return L2SqrSIMD16ResSSE(x, y, dimension);
  }
};
#endif

} // namespace flatnav