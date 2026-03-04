#pragma once

#include <flatnav/util/SimdUtils.h>

namespace flatnav::util {

#if defined(USE_AVX512)
static float compute_l2_avx512(const void* x, const void* y, const size_t& dimension) {
  const float* pointer_x = static_cast<const float*>(x);
  const float* pointer_y = static_cast<const float*>(y);

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

// Computes the squared L2 distance between two uint8 vectors using AVX512BW.
//
// Strategy: load 32 uint8 values (256 bits) per iteration and zero-extend
// to 32 int16 values in a 512-bit register via _mm512_cvtepu8_epi16
// (AVX512BW). After zero-extending both vectors, we subtract in int16
// and use _mm512_madd_epi16(diff, diff) to square-and-accumulate into int32.
//
// Handles arbitrary dimensions via a scalar tail loop.
static float compute_l2_avx512_uint8(const void* x, const void* y, const size_t& dimension) {
  const uint8_t* pointer_x = static_cast<const uint8_t*>(x);
  const uint8_t* pointer_y = static_cast<const uint8_t*>(y);

  __m512i sum = _mm512_setzero_si512();
  size_t aligned_dimension = dimension & ~0x1F;  // round down to multiple of 32
  size_t i = 0;

  for (; i < aligned_dimension; i += 32) {
    // Load 32 uint8 values and zero-extend to 32 int16 values
    __m512i a_i16 = _mm512_cvtepu8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pointer_x + i)));
    __m512i b_i16 = _mm512_cvtepu8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pointer_y + i)));

    __m512i diff = _mm512_sub_epi16(a_i16, b_i16);
    sum = _mm512_add_epi32(sum, _mm512_madd_epi16(diff, diff));
  }

  // Scalar tail for remaining elements
  int32_t partial_sum = 0;
  for (; i < dimension; i++) {
    int diff = static_cast<int>(pointer_x[i]) - static_cast<int>(pointer_y[i]);
    partial_sum += diff * diff;
  }

  int32_t total = _mm512_reduce_add_epi32(sum);
  return static_cast<float>(total + partial_sum);
}

// Computes the squared L2 distance between two int8 vectors using AVX512BW.
//
// Strategy: load 32 int8 values (256 bits) per iteration and sign-extend the
// entire chunk to 32 int16 values in a 512-bit register via
// _mm512_cvtepi8_epi16 (AVX512BW). This is simpler than loading a full 64-byte
// register and splitting it — we avoid the extract/cast dance and the 512-bit
// int8 subtract entirely.
//
// After sign-extending both vectors, we subtract in int16 and use
// _mm512_madd_epi16(diff, diff) to square adjacent pairs and accumulate into
// int32. This produces 16 int32 lanes per iteration.
//
// Note: on VNNI-capable CPUs, _mm512_dpwssd_epi32 could fuse the madd+add
// into one instruction. We use the equivalent madd+add sequence which works
// on all AVX512 hardware, not just VNNI-capable CPUs.
//
// Handles arbitrary dimensions via a scalar tail loop.
static float compute_l2_avx512_int8(const void* x, const void* y, const size_t& dimension) {
  const int8_t* pointer_x = static_cast<const int8_t*>(x);
  const int8_t* pointer_y = static_cast<const int8_t*>(y);

  __m512i sum = _mm512_setzero_si512();
  size_t aligned_dimension = dimension & ~0x1F;  // round down to multiple of 32
  size_t i = 0;

  for (; i < aligned_dimension; i += 32) {
    // Load 32 int8 values (256 bits) and sign-extend to 32 int16 values (512 bits)
    __m512i a_i16 = _mm512_cvtepi8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pointer_x + i)));
    __m512i b_i16 = _mm512_cvtepi8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pointer_y + i)));

    // Subtract in int16 (no overflow: int16 range [-32768, 32767] handles
    // max diff of 255 from int8 subtraction)
    __m512i diff = _mm512_sub_epi16(a_i16, b_i16);

    // Square and accumulate: madd multiplies adjacent int16 pairs and sums
    // them into int32: (d[0]^2 + d[1]^2, d[2]^2 + d[3]^2, ...)
    sum = _mm512_add_epi32(sum, _mm512_madd_epi16(diff, diff));
  }

  // Scalar tail for remaining elements
  int32_t partial_sum = 0;
  for (; i < dimension; i++) {
    int diff = pointer_x[i] - pointer_y[i];
    partial_sum += diff * diff;
  }

  // Horizontal reduction of all 16 int32 lanes to a single scalar
  int32_t total = _mm512_reduce_add_epi32(sum);
  return static_cast<float>(total + partial_sum);
}

#endif  // USE_AVX512

#if defined(USE_AVX)

static float compute_l2_avx2(const void* x, const void* y, const size_t& dimension) {
  const float* pointer_x = static_cast<const float*>(x);
  const float* pointer_y = static_cast<const float*>(y);

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

// Computes the squared L2 distance between two int8 vectors using AVX2.
//
// Strategy: load 32 int8 values per iteration, split each 256-bit register
// into two 128-bit halves, and sign-extend each half from 16 x int8 to
// 16 x int16 in a 256-bit register using _mm256_cvtepi8_epi16 (AVX2). We
// then subtract in int16 and use _mm256_madd_epi16(diff, diff) to square
// adjacent pairs and accumulate into int32.
//
// Why sign-extend first, then subtract (instead of _mm256_sub_epi8 then split)?
// _mm256_sub_epi8 wraps around when the true difference exceeds int8 range
// (e.g., 127 - (-128) = 255 wraps to -1). Extending to int16 first avoids
// this overflow entirely.
//
// Two separate int32 accumulators (low/high) are maintained to exploit
// instruction-level parallelism — the CPU can execute independent add chains
// in parallel.
//
// Handles arbitrary dimensions via a scalar tail loop.
static float compute_l2_avx2_int8(const void* x, const void* y, const size_t& dimension) {
  const int8_t* pointer_x = static_cast<const int8_t*>(x);
  const int8_t* pointer_y = static_cast<const int8_t*>(y);

  __m256i sum_lo = _mm256_setzero_si256();
  __m256i sum_hi = _mm256_setzero_si256();
  size_t aligned_dimension = dimension & ~0x1F;  // round down to multiple of 32
  size_t i = 0;

  for (; i < aligned_dimension; i += 32) {
    __m256i a_i8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pointer_x + i));
    __m256i b_i8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pointer_y + i));

    // Sign-extend low 16 bytes (128 bits) to 16 x int16 (256 bits)
    __m256i a_i16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a_i8));
    __m256i b_i16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_i8));

    // Sign-extend high 16 bytes to 16 x int16
    __m256i a_i16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8, 1));
    __m256i b_i16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8, 1));

    // Subtract in int16, then square-and-accumulate into int32
    __m256i diff_lo = _mm256_sub_epi16(a_i16_lo, b_i16_lo);
    __m256i diff_hi = _mm256_sub_epi16(a_i16_hi, b_i16_hi);

    sum_lo = _mm256_add_epi32(sum_lo, _mm256_madd_epi16(diff_lo, diff_lo));
    sum_hi = _mm256_add_epi32(sum_hi, _mm256_madd_epi16(diff_hi, diff_hi));
  }

  // Scalar tail for remaining elements
  int32_t partial_sum = 0;
  for (; i < dimension; i++) {
    int diff = pointer_x[i] - pointer_y[i];
    partial_sum += diff * diff;
  }

  // Merge the two accumulators and horizontally reduce
  __m256i combined = _mm256_add_epi32(sum_lo, sum_hi);
  __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(combined),
                                  _mm256_extracti128_si256(combined, 1));
  sum128 = _mm_hadd_epi32(sum128, sum128);
  sum128 = _mm_hadd_epi32(sum128, sum128);
  return static_cast<float>(_mm_cvtsi128_si32(sum128) + partial_sum);
}

// Computes the squared L2 distance between two uint8 vectors using AVX2.
//
// Strategy: load 32 uint8 values per iteration, split each 256-bit register
// into two 128-bit halves, and zero-extend each half from 16 x uint8 to
// 16 x int16 in a 256-bit register using _mm256_cvtepu8_epi16 (AVX2).
// After zero-extending both vectors, we subtract in int16 and use
// _mm256_madd_epi16(diff, diff) to square-and-accumulate into int32.
//
// Two separate int32 accumulators (low/high) exploit instruction-level
// parallelism. Handles arbitrary dimensions via a scalar tail loop.
static float compute_l2_avx2_uint8(const void* x, const void* y, const size_t& dimension) {
  const uint8_t* pointer_x = static_cast<const uint8_t*>(x);
  const uint8_t* pointer_y = static_cast<const uint8_t*>(y);

  __m256i sum_lo = _mm256_setzero_si256();
  __m256i sum_hi = _mm256_setzero_si256();
  size_t aligned_dimension = dimension & ~0x1F;  // round down to multiple of 32
  size_t i = 0;

  for (; i < aligned_dimension; i += 32) {
    __m256i a_u8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pointer_x + i));
    __m256i b_u8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pointer_y + i));

    // Zero-extend low 16 bytes to 16 x int16
    __m256i a_i16_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_u8));
    __m256i b_i16_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_u8));

    // Zero-extend high 16 bytes to 16 x int16
    __m256i a_i16_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a_u8, 1));
    __m256i b_i16_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_u8, 1));

    // Subtract in int16, then square-and-accumulate into int32
    __m256i diff_lo = _mm256_sub_epi16(a_i16_lo, b_i16_lo);
    __m256i diff_hi = _mm256_sub_epi16(a_i16_hi, b_i16_hi);

    sum_lo = _mm256_add_epi32(sum_lo, _mm256_madd_epi16(diff_lo, diff_lo));
    sum_hi = _mm256_add_epi32(sum_hi, _mm256_madd_epi16(diff_hi, diff_hi));
  }

  // Scalar tail for remaining elements
  int32_t partial_sum = 0;
  for (; i < dimension; i++) {
    int diff = static_cast<int>(pointer_x[i]) - static_cast<int>(pointer_y[i]);
    partial_sum += diff * diff;
  }

  // Merge the two accumulators and horizontally reduce
  __m256i combined = _mm256_add_epi32(sum_lo, sum_hi);
  __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(combined),
                                  _mm256_extracti128_si256(combined, 1));
  sum128 = _mm_hadd_epi32(sum128, sum128);
  sum128 = _mm_hadd_epi32(sum128, sum128);
  return static_cast<float>(_mm_cvtsi128_si32(sum128) + partial_sum);
}

#endif  // USE_AVX

#if defined(USE_SSE)

static float compute_l2_sse(const void* x, const void* y, const size_t& dimension) {
  const float* pointer_x = static_cast<const float*>(x);
  const float* pointer_y = static_cast<const float*>(y);

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

// Computes the squared L2 distance between two int8 vectors using SSE4.1.
//
// Strategy: load 16 int8 values per iteration from each vector, then
// sign-extend each 8-byte half from int8 to int16 BEFORE subtracting.
// This avoids overflow from _mm_sub_epi8: int8 subtraction wraps around
// when the true difference exceeds [-128, 127] (e.g., 127 - (-128) = 255
// wraps to -1 in int8). By extending to int16 first and subtracting there,
// the full difference range [-255, 255] fits safely.
//
// _mm_cvtepi8_epi16 (SSE4.1) only sign-extends the LOW 8 bytes of a 128-bit
// register. We process the high 8 bytes by shifting them down with
// _mm_srli_si128(v, 8) before sign-extending.
//
// After subtraction in int16, _mm_madd_epi16(diff, diff) squares adjacent
// pairs and sums them into int32: (d[0]^2 + d[1]^2, d[2]^2 + d[3]^2, ...).
//
// Accumulation safety: max |diff| = 255, max diff^2 = 65025, and madd produces
// at most 2 * 65025 = 130050 per int32 lane per iteration. int32 overflow
// requires dimension > ~16500, which is well beyond typical int8 vector sizes.
static float compute_l2_sse_int8(const void* x, const void* y, const size_t& dimension) {
  const int8_t* pointer_x = static_cast<const int8_t*>(x);
  const int8_t* pointer_y = static_cast<const int8_t*>(y);

  __m128i sum = _mm_setzero_si128();
  size_t aligned_dimension = dimension & ~0xF;
  size_t i = 0;

  for (; i < aligned_dimension; i += 16) {
    __m128i vx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pointer_x + i));
    __m128i vy = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pointer_y + i));

    // Low 8 bytes: sign-extend int8 → int16, subtract, then square-and-accumulate
    __m128i x_lo = _mm_cvtepi8_epi16(vx);
    __m128i y_lo = _mm_cvtepi8_epi16(vy);
    __m128i diff_lo = _mm_sub_epi16(x_lo, y_lo);
    sum = _mm_add_epi32(sum, _mm_madd_epi16(diff_lo, diff_lo));

    // High 8 bytes: shift down, sign-extend, subtract, then square-and-accumulate
    __m128i x_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vx, 8));
    __m128i y_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vy, 8));
    __m128i diff_hi = _mm_sub_epi16(x_hi, y_hi);
    sum = _mm_add_epi32(sum, _mm_madd_epi16(diff_hi, diff_hi));
  }

  // Scalar tail for remaining elements not divisible by 16
  int32_t partial_sum = 0;
  for (; i < dimension; i++) {
    int diff = pointer_x[i] - pointer_y[i];
    partial_sum += diff * diff;
  }

  // Horizontal reduction: sum the 4 int32 lanes
  int32_t buffer[4];
  _mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), sum);
  return static_cast<float>(buffer[0] + buffer[1] + buffer[2] + buffer[3] + partial_sum);
}

// Computes the squared L2 distance between two uint8 vectors using SSE4.1.
//
// Strategy: load 16 uint8 values per iteration, zero-extend each 8-byte half
// from uint8 to int16 via _mm_cvtepu8_epi16 (SSE4.1). After zero-extending
// both vectors, we subtract in int16 and use _mm_madd_epi16(diff, diff) to
// square-and-accumulate into int32.
//
// _mm_cvtepu8_epi16 only zero-extends the LOW 8 bytes of a 128-bit register.
// We process the high 8 bytes by shifting them down with _mm_srli_si128(v, 8)
// before zero-extending.
//
// Handles arbitrary dimensions via a scalar tail loop.
static float compute_l2_sse_uint8(const void* x, const void* y, const size_t& dimension) {
  const uint8_t* pointer_x = static_cast<const uint8_t*>(x);
  const uint8_t* pointer_y = static_cast<const uint8_t*>(y);

  __m128i sum = _mm_setzero_si128();
  size_t aligned_dimension = dimension & ~0xF;  // round down to multiple of 16
  size_t i = 0;

  for (; i < aligned_dimension; i += 16) {
    __m128i vx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pointer_x + i));
    __m128i vy = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pointer_y + i));

    // Low 8 bytes: zero-extend uint8 → int16, subtract, then square-and-accumulate
    __m128i x_lo = _mm_cvtepu8_epi16(vx);
    __m128i y_lo = _mm_cvtepu8_epi16(vy);
    __m128i diff_lo = _mm_sub_epi16(x_lo, y_lo);
    sum = _mm_add_epi32(sum, _mm_madd_epi16(diff_lo, diff_lo));

    // High 8 bytes: shift down, zero-extend, subtract, then square-and-accumulate
    __m128i x_hi = _mm_cvtepu8_epi16(_mm_srli_si128(vx, 8));
    __m128i y_hi = _mm_cvtepu8_epi16(_mm_srli_si128(vy, 8));
    __m128i diff_hi = _mm_sub_epi16(x_hi, y_hi);
    sum = _mm_add_epi32(sum, _mm_madd_epi16(diff_hi, diff_hi));
  }

  // Scalar tail for remaining elements not divisible by 16
  int32_t partial_sum = 0;
  for (; i < dimension; i++) {
    int diff = static_cast<int>(pointer_x[i]) - static_cast<int>(pointer_y[i]);
    partial_sum += diff * diff;
  }

  // Horizontal reduction: sum the 4 int32 lanes
  int32_t buffer[4];
  _mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), sum);
  return static_cast<float>(buffer[0] + buffer[1] + buffer[2] + buffer[3] + partial_sum);
}

#endif  // USE_SSE4_1

static float compute_l2_sse_4aligned(const void* x, const void* y, const size_t& dimension) {
  const float* pointer_x = static_cast<const float*>(x);
  const float* pointer_y = static_cast<const float*>(y);

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

static float compute_l2_sse_residual_16(const void* x, const void* y, const size_t& dimension) {

  size_t dimension_aligned = dimension >> 4 << 4;
  float aligned_distance = compute_l2_sse(x, y, dimension_aligned);
  size_t residual_dimension = dimension - dimension_aligned;
  float residual_distance = 0.0f;
  const float* pointer_x = static_cast<const float*>(x) + dimension_aligned;
  const float* pointer_y = static_cast<const float*>(y) + dimension_aligned;
  for (size_t i = 0; i < residual_dimension; i++) {
    float difference = *pointer_x - *pointer_y;
    residual_distance += difference * difference;
    pointer_x++;
    pointer_y++;
  }
  return aligned_distance + residual_distance;
}

static float compute_l2_sse_residual_4(const void* x, const void* y, const size_t& dimension) {
  size_t dimension_aligned = dimension >> 2 << 2;
  float aligned_distance = compute_l2_sse_4aligned(x, y, dimension_aligned);
  size_t residual_dimension = dimension - dimension_aligned;
  float residual_distance = 0.0f;
  const float* pointer_x = static_cast<const float*>(x) + dimension_aligned;
  const float* pointer_y = static_cast<const float*>(y) + dimension_aligned;
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