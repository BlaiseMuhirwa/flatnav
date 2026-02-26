#pragma once

#include <flatnav/util/SimdUtils.h>

namespace flatnav::util {

#if defined(USE_AVX512)

static float compute_ip_avx512(const void* x, const void* y, const size_t& dimension) {
  const float* pointer_x = static_cast<const float*>(x);
  const float* pointer_y = static_cast<const float*>(y);

  // Align to 16-floats boundary
  const float* end_x = pointer_x + (dimension >> 4 << 4);
  simd16float32 product, v1, v2;

  simd16float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    product = v1 * v2;
    sum += product;
    pointer_x += 16;
    pointer_y += 16;
  }
  float total = sum.reduce_add();
  return 1.0f - total;
}

// Computes the inner product distance (1 - dot) between two int8 vectors using
// AVX512BW.
//
// Strategy: load 32 int8 values (256 bits)
// per iteration and sign-extend to 32 int16 values in a 512-bit register via
// _mm512_cvtepi8_epi16 (AVX512BW). This avoids loading a full 64-byte register
// and splitting — instead we go directly from 256-bit int8 to 512-bit int16.
//
// The dot product is accumulated using _mm512_madd_epi16(a, b), which multiplies
// adjacent int16 pairs and sums them into int32: (a[0]*b[0] + a[1]*b[1], ...).
//
// Why not _mm512_dpbusd_epi32 (VNNI byte dot product)? That instruction is
// asymmetric — it treats one operand as unsigned and the other as signed.
// For two signed int8 vectors, this doesn't work correctly (the -128 edge case
// cannot be negated in int8). We use the symmetric madd_epi16 approach instead.
//
// Returns 1.0f - dot_product to match the inner product distance convention.
static float compute_ip_avx512_int8(const void* x, const void* y, const size_t& dimension) {
  const int8_t* pointer_x = static_cast<const int8_t*>(x);
  const int8_t* pointer_y = static_cast<const int8_t*>(y);

  __m512i sum = _mm512_setzero_si512();
  size_t aligned_dimension = dimension & ~0x1F;  // round down to multiple of 32
  size_t i = 0;

  for (; i < aligned_dimension; i += 32) {
    // Load 32 int8 values and sign-extend to 32 int16 values
    __m512i a_i16 = _mm512_cvtepi8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pointer_x + i)));
    __m512i b_i16 = _mm512_cvtepi8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pointer_y + i)));

    // Multiply-and-add: (a[0]*b[0] + a[1]*b[1], a[2]*b[2] + a[3]*b[3], ...)
    // Each pair of int16 products is summed into one int32 lane.
    sum = _mm512_add_epi32(sum, _mm512_madd_epi16(a_i16, b_i16));
  }

  // Scalar tail for remaining elements
  int32_t partial_sum = 0;
  for (; i < dimension; i++) {
    partial_sum += static_cast<int32_t>(pointer_x[i]) * static_cast<int32_t>(pointer_y[i]);
  }

  int32_t total = _mm512_reduce_add_epi32(sum);
  float dot = static_cast<float>(total + partial_sum);
  return 1.0f - dot;
}

#endif  // USE_AVX512

#if defined(USE_AVX)
static float compute_ip_avx(const void* x, const void* y, const size_t& dimension) {
  const float* pointer_x = static_cast<const float*>(x);
  const float* pointer_y = static_cast<const float*>(y);

  const float* end_x = pointer_x + (dimension >> 4 << 4);
  simd8float32 product, v1, v2;
  simd8float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    product = v1 * v2;
    sum += product;
    pointer_x += 8;
    pointer_y += 8;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    product = v1 * v2;
    sum += product;
    pointer_x += 8;
    pointer_y += 8;
  }

  float total = sum.reduce_add();
  return 1.0f - total;
}

static float compute_ip_avx_4aligned(const void* x, const void* y, const size_t& dimension) {

  const float* pointer_x = static_cast<const float*>(x);
  const float* pointer_y = static_cast<const float*>(y);

  const float* first_chunk_end = pointer_x + (dimension >> 4 << 4);
  const float* second_chunk_end = pointer_x + (dimension >> 2 << 2);

  simd8float32 v1, v2;
  simd8float32 sum(0.0f);

  while (pointer_x != first_chunk_end) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 8;
    pointer_y += 8;
  }

  // TODO: See if we can reduce this to fewer instructions
  simd4float32 aggregate = simd4float32(sum.get_low()) + simd4float32(sum.get_high());
  simd4float32 v1_residual, v2_residual;

  while (pointer_x != second_chunk_end) {
    v1_residual.loadu(pointer_x);
    v2_residual.loadu(pointer_y);
    aggregate += (v1_residual * v2_residual);
    pointer_x += 4;
    pointer_y += 4;
  }

  float total = aggregate.reduce_add();
  return 1.0f - total;
}

// Computes the inner product distance (1 - dot) between two int8 vectors using
// AVX2.
//
// Strategy: load 32 int8 values per
// iteration, split each 256-bit register into its two 128-bit halves, and
// sign-extend each half from 16 x int8 to 16 x int16 in a 256-bit register
// using _mm256_cvtepi8_epi16 (AVX2).
//
// The dot product is accumulated via _mm256_madd_epi16(a_half, b_half), which
// multiplies adjacent int16 pairs and horizontally sums them into int32:
//   (a[0]*b[0] + a[1]*b[1], a[2]*b[2] + a[3]*b[3], ...)
//
// Why not use _mm256_maddubs_epi16? That instruction expects one unsigned and
// one signed operand. For two signed int8 vectors, you'd need to take abs() of
// one and flip the sign of the other — but int8 value -128 cannot be negated
// (it stays -128 due to two's complement), causing incorrect results.
//
// Two separate int32 accumulators (low/high) exploit instruction-level
// parallelism — the CPU can retire independent addition chains in parallel.
//
// Returns 1.0f - dot_product to match the inner product distance convention.
static float compute_ip_avx2_int8(const void* x, const void* y, const size_t& dimension) {
  const int8_t* pointer_x = static_cast<const int8_t*>(x);
  const int8_t* pointer_y = static_cast<const int8_t*>(y);

  __m256i sum_lo = _mm256_setzero_si256();
  __m256i sum_hi = _mm256_setzero_si256();
  size_t aligned_dimension = dimension & ~0x1F;  // round down to multiple of 32
  size_t i = 0;

  for (; i < aligned_dimension; i += 32) {
    __m256i a_i8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pointer_x + i));
    __m256i b_i8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pointer_y + i));

    // Sign-extend low 16 bytes to 16 x int16
    __m256i a_i16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a_i8));
    __m256i b_i16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_i8));

    // Sign-extend high 16 bytes to 16 x int16
    __m256i a_i16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8, 1));
    __m256i b_i16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8, 1));

    // Multiply-and-add into int32 accumulators
    sum_lo = _mm256_add_epi32(sum_lo, _mm256_madd_epi16(a_i16_lo, b_i16_lo));
    sum_hi = _mm256_add_epi32(sum_hi, _mm256_madd_epi16(a_i16_hi, b_i16_hi));
  }

  // Scalar tail for remaining elements
  int32_t partial_sum = 0;
  for (; i < dimension; i++) {
    partial_sum += static_cast<int32_t>(pointer_x[i]) * static_cast<int32_t>(pointer_y[i]);
  }

  // Merge accumulators and horizontally reduce
  __m256i combined = _mm256_add_epi32(sum_lo, sum_hi);
  __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(combined),
                                  _mm256_extracti128_si256(combined, 1));
  sum128 = _mm_hadd_epi32(sum128, sum128);
  sum128 = _mm_hadd_epi32(sum128, sum128);
  float dot = static_cast<float>(_mm_cvtsi128_si32(sum128) + partial_sum);
  return 1.0f - dot;
}

#endif  // USE_AVX

#if defined(USE_SSE)

static float compute_ip_sse(const void* x, const void* y, const size_t& dimension) {
  const float* pointer_x = static_cast<const float*>(x);
  const float* pointer_y = static_cast<const float*>(y);

  const float* end_x = pointer_x + (dimension >> 4 << 4);
  simd4float32 v1, v2;
  simd4float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;
  }

  float total = sum.reduce_add();
  return 1.0f - total;
}

static float compute_ip_sse_4aligned(const void* x, const void* y, const size_t& dimension) {
  const float* pointer_x = static_cast<const float*>(x);
  const float* pointer_y = static_cast<const float*>(y);
  const float* first_chunk_end = pointer_x + (dimension >> 4 << 4);
  const float* second_chunk_end = pointer_x + (dimension >> 2 << 2);

  simd4float32 v1, v2;
  simd4float32 sum(0.0f);
  while (pointer_x != first_chunk_end) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;
  }

  while (pointer_x != second_chunk_end) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;
  }

  float total = sum.reduce_add();
  return 1.0f - total;
}

#if defined(USE_SSE4_1)

// Computes the inner product distance (1 - dot) between two int8 vectors using
// SSE4.1.
//
// Strategy: load 16 int8 values per iteration from each vector. Since
// _mm_cvtepi8_epi16 (SSE4.1) only sign-extends the LOW 8 bytes of a 128-bit
// register, we must process each 16-byte load in two halves:
//   - Low 8 bytes: _mm_cvtepi8_epi16(v) directly
//   - High 8 bytes: _mm_cvtepi8_epi16(_mm_srli_si128(v, 8)) — shift high bytes
//     down first
//
// For each half, _mm_madd_epi16(a_half, b_half) multiplies adjacent int16 pairs
// and sums them into int32: (a[0]*b[0] + a[1]*b[1], a[2]*b[2] + a[3]*b[3]).
//
// Unlike L2, we cannot subtract first and square — for inner product we need
// both vectors independently sign-extended before cross-multiplication.
//
// Returns 1.0f - dot_product to match the inner product distance convention.
static float compute_ip_sse_int8(const void* x, const void* y, const size_t& dimension) {
  const int8_t* pointer_x = static_cast<const int8_t*>(x);
  const int8_t* pointer_y = static_cast<const int8_t*>(y);

  __m128i sum = _mm_setzero_si128();
  size_t aligned_dimension = dimension & ~0xF;  // round down to multiple of 16
  size_t i = 0;

  for (; i < aligned_dimension; i += 16) {
    __m128i vx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pointer_x + i));
    __m128i vy = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pointer_y + i));

    // Low 8 bytes: sign-extend int8 → int16, then multiply-and-add
    __m128i x_lo = _mm_cvtepi8_epi16(vx);
    __m128i y_lo = _mm_cvtepi8_epi16(vy);
    sum = _mm_add_epi32(sum, _mm_madd_epi16(x_lo, y_lo));

    // High 8 bytes: shift down, sign-extend, then multiply-and-add
    __m128i x_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vx, 8));
    __m128i y_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vy, 8));
    sum = _mm_add_epi32(sum, _mm_madd_epi16(x_hi, y_hi));
  }

  // Scalar tail for remaining elements
  int32_t partial_sum = 0;
  for (; i < dimension; i++) {
    partial_sum += static_cast<int32_t>(pointer_x[i]) * static_cast<int32_t>(pointer_y[i]);
  }

  // Horizontal reduction: sum the 4 int32 lanes
  int32_t buffer[4];
  _mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), sum);
  float dot = static_cast<float>(buffer[0] + buffer[1] + buffer[2] + buffer[3] + partial_sum);
  return 1.0f - dot;
}

#endif  // USE_SSE4_1

static float compute_ip_sse_residual_16(const void* x, const void* y, const size_t& dimension) {
  size_t aligned_dimension = dimension >> 4 << 4;
  size_t residual_dimension = dimension - aligned_dimension;

  // We need to subtract 1.0f from the result, and then multiply by -1.0f
  // in order to get the actual dot product.
  float first_chunk_sum = compute_ip_sse(x, y, aligned_dimension);
  first_chunk_sum -= 1.0f;
  first_chunk_sum *= -1.0f;

  float residual_sum = 0.0f;
  const float* pointer_x = static_cast<const float*>(x) + aligned_dimension;
  const float* pointer_y = static_cast<const float*>(y) + aligned_dimension;
  for (size_t i = 0; i < residual_dimension; i++) {
    residual_sum += pointer_x[i] * pointer_y[i];
  }
  return 1.0f - (first_chunk_sum + residual_sum);
}

static float compute_ip_sse_residual_4(const void* x, const void* y, const size_t& dimension) {
  size_t aligned_dimension = dimension >> 2 << 2;
  size_t residual_dimension = dimension - aligned_dimension;

  // We need to subtract 1.0f from the result, and then multiply by -1.0f
  // in order to get the actual dot product.
  float first_chunk_sum = compute_ip_sse_4aligned(x, y, aligned_dimension);
  first_chunk_sum -= 1.0f;
  first_chunk_sum *= -1.0f;

  float residual_sum = 0.0f;
  const float* pointer_x = static_cast<const float*>(x) + aligned_dimension;
  const float* pointer_y = static_cast<const float*>(y) + aligned_dimension;
  for (size_t i = 0; i < residual_dimension; i++) {
    residual_sum += pointer_x[i] * pointer_y[i];
  }
  return 1.0f - (first_chunk_sum + residual_sum);
}

#endif  // USE_SSE

}  // namespace flatnav::util
