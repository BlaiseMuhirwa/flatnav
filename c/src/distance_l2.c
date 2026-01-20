/**
 * @file distance_l2.c
 * @brief Squared L2 (Euclidean) distance implementations
 */

#include <flatnav/flatnav.h>
#include <stdlib.h>
#include <string.h>

/*============================================================================
 * Scalar Implementations (Always Available)
 *============================================================================*/

static float fn__l2_scalar_f32(const float* x, const float* y, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; i++) {
    float diff = x[i] - y[i];
    sum += diff * diff;
  }
  return sum;
}

static float fn__l2_scalar_i8(const int8_t* x, const int8_t* y, uint32_t dim) {
  int32_t sum = 0;
  for (uint32_t i = 0; i < dim; i++) {
    int32_t diff = (int32_t)x[i] - (int32_t)y[i];
    sum += diff * diff;
  }
  return (float)sum;
}

static float fn__l2_scalar_u8(const uint8_t* x, const uint8_t* y, uint32_t dim) {
  int32_t sum = 0;
  for (uint32_t i = 0; i < dim; i++) {
    int32_t diff = (int32_t)x[i] - (int32_t)y[i];
    sum += diff * diff;
  }
  return (float)sum;
}

/*============================================================================
 * SSE Implementations
 *============================================================================*/

#ifdef FN_USE_SSE
#include <emmintrin.h>
#include <xmmintrin.h>

__attribute__((target("sse"))) static float fn__l2_sse_f32(const float* x, const float* y, uint32_t dim) {
  __m128 sum = _mm_setzero_ps();
  uint32_t aligned_dim = dim & ~3u;

  for (uint32_t i = 0; i < aligned_dim; i += 4) {
    __m128 vx = _mm_loadu_ps(x + i);
    __m128 vy = _mm_loadu_ps(y + i);
    __m128 diff = _mm_sub_ps(vx, vy);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
  }

  /* Horizontal sum */
  __m128 shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
  sum = _mm_add_ps(sum, shuf);
  shuf = _mm_movehl_ps(shuf, sum);
  sum = _mm_add_ss(sum, shuf);

  float result = _mm_cvtss_f32(sum);

  /* Handle residual */
  for (uint32_t i = aligned_dim; i < dim; i++) {
    float diff = x[i] - y[i];
    result += diff * diff;
  }

  return result;
}

#endif /* FN_USE_SSE */

/*============================================================================
 * SSE4.1 Implementations
 *============================================================================*/

#ifdef FN_USE_SSE4
#include <smmintrin.h>

__attribute__((target("sse4.1"))) static float fn__l2_sse4_i8(const int8_t* x, const int8_t* y,
                                                              uint32_t dim) {
  __m128i sum = _mm_setzero_si128();
  uint32_t aligned_dim = dim & ~15u;

  for (uint32_t i = 0; i < aligned_dim; i += 16) {
    __m128i vx = _mm_loadu_si128((const __m128i*)(x + i));
    __m128i vy = _mm_loadu_si128((const __m128i*)(y + i));

    /* Compute absolute difference using saturating subtraction */
    __m128i diff1 = _mm_subs_epi8(vx, vy);
    __m128i diff2 = _mm_subs_epi8(vy, vx);
    __m128i absdiff = _mm_or_si128(diff1, diff2);

    /* Unpack to 16-bit and square */
    __m128i lo = _mm_unpacklo_epi8(absdiff, _mm_setzero_si128());
    __m128i hi = _mm_unpackhi_epi8(absdiff, _mm_setzero_si128());

    __m128i sq_lo = _mm_mullo_epi16(lo, lo);
    __m128i sq_hi = _mm_mullo_epi16(hi, hi);

    /* Accumulate to 32-bit */
    sum = _mm_add_epi32(sum, _mm_unpacklo_epi16(sq_lo, _mm_setzero_si128()));
    sum = _mm_add_epi32(sum, _mm_unpackhi_epi16(sq_lo, _mm_setzero_si128()));
    sum = _mm_add_epi32(sum, _mm_unpacklo_epi16(sq_hi, _mm_setzero_si128()));
    sum = _mm_add_epi32(sum, _mm_unpackhi_epi16(sq_hi, _mm_setzero_si128()));
  }

  /* Horizontal sum */
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  int32_t result = _mm_cvtsi128_si32(sum);

  /* Handle residual */
  for (uint32_t i = aligned_dim; i < dim; i++) {
    int32_t diff = (int32_t)x[i] - (int32_t)y[i];
    result += diff * diff;
  }

  return (float)result;
}

#endif /* FN_USE_SSE4 */

/*============================================================================
 * AVX2 Implementations
 *============================================================================*/

#ifdef FN_USE_AVX2
#include <immintrin.h>

__attribute__((target("avx2,fma"))) static float fn__l2_avx2_f32(const float* x, const float* y,
                                                                 uint32_t dim) {
  __m256 sum = _mm256_setzero_ps();
  uint32_t aligned_dim = dim & ~7u;

  for (uint32_t i = 0; i < aligned_dim; i += 8) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = _mm256_loadu_ps(y + i);
    __m256 diff = _mm256_sub_ps(vx, vy);
    sum = _mm256_fmadd_ps(diff, diff, sum);
  }

  /* Horizontal sum (256-bit) */
  __m128 low = _mm256_castps256_ps128(sum);
  __m128 high = _mm256_extractf128_ps(sum, 1);
  __m128 sum128 = _mm_add_ps(low, high);

  __m128 shuf = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2, 3, 0, 1));
  sum128 = _mm_add_ps(sum128, shuf);
  shuf = _mm_movehl_ps(shuf, sum128);
  sum128 = _mm_add_ss(sum128, shuf);

  float result = _mm_cvtss_f32(sum128);

  /* Handle residual */
  for (uint32_t i = aligned_dim; i < dim; i++) {
    float diff = x[i] - y[i];
    result += diff * diff;
  }

  return result;
}

__attribute__((target("avx2"))) static float fn__l2_avx2_i8(const int8_t* x, const int8_t* y, uint32_t dim) {
  __m256i sum = _mm256_setzero_si256();
  uint32_t aligned_dim = dim & ~31u;

  for (uint32_t i = 0; i < aligned_dim; i += 32) {
    __m256i vx = _mm256_loadu_si256((const __m256i*)(x + i));
    __m256i vy = _mm256_loadu_si256((const __m256i*)(y + i));

    /* Sign-extend to 16-bit for subtraction */
    __m256i x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vx));
    __m256i x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx, 1));
    __m256i y_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vy));
    __m256i y_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vy, 1));

    __m256i diff_lo = _mm256_sub_epi16(x_lo, y_lo);
    __m256i diff_hi = _mm256_sub_epi16(x_hi, y_hi);

    /* Square and accumulate using madd (a*a + b*b) */
    __m256i sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
    __m256i sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);

    sum = _mm256_add_epi32(sum, sq_lo);
    sum = _mm256_add_epi32(sum, sq_hi);
  }

  /* Horizontal sum */
  __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
  sum128 = _mm_hadd_epi32(sum128, sum128);
  sum128 = _mm_hadd_epi32(sum128, sum128);
  int32_t result = _mm_cvtsi128_si32(sum128);

  /* Handle residual */
  for (uint32_t i = aligned_dim; i < dim; i++) {
    int32_t diff = (int32_t)x[i] - (int32_t)y[i];
    result += diff * diff;
  }

  return (float)result;
}

#endif /* FN_USE_AVX2 */

/*============================================================================
 * AVX-512 Implementations
 *============================================================================*/

#ifdef FN_USE_AVX512
#include <immintrin.h>

__attribute__((target("avx512f"))) static float fn__l2_avx512_f32(const float* x, const float* y,
                                                                  uint32_t dim) {
  __m512 sum = _mm512_setzero_ps();
  uint32_t aligned_dim = dim & ~15u;

  for (uint32_t i = 0; i < aligned_dim; i += 16) {
    __m512 vx = _mm512_loadu_ps(x + i);
    __m512 vy = _mm512_loadu_ps(y + i);
    __m512 diff = _mm512_sub_ps(vx, vy);
    sum = _mm512_fmadd_ps(diff, diff, sum);
  }

  float result = _mm512_reduce_add_ps(sum);

  /* Handle residual */
  for (uint32_t i = aligned_dim; i < dim; i++) {
    float diff = x[i] - y[i];
    result += diff * diff;
  }

  return result;
}

#endif /* FN_USE_AVX512 */

/*============================================================================
 * Distance Function Dispatch
 *============================================================================*/

/* Function pointer types for internal dispatch */
typedef float (*fn__l2_f32_func_t)(const float*, const float*, uint32_t);
typedef float (*fn__l2_i8_func_t)(const int8_t*, const int8_t*, uint32_t);
typedef float (*fn__l2_u8_func_t)(const uint8_t*, const uint8_t*, uint32_t);

/* Global dispatch table */
static fn__l2_f32_func_t g_l2_f32 = fn__l2_scalar_f32;
static fn__l2_i8_func_t g_l2_i8 = fn__l2_scalar_i8;
static fn__l2_u8_func_t g_l2_u8 = fn__l2_scalar_u8;

#ifndef FN_DPU_TARGET
void fn__init_l2_dispatch(void) {
  /* Select best available implementation */
#ifdef FN_USE_AVX512
  if (fn_cpu_supports_avx512()) {
    g_l2_f32 = fn__l2_avx512_f32;
  }
#endif

#ifdef FN_USE_AVX2
  if (fn_cpu_supports_avx2()) {
    if (g_l2_f32 == fn__l2_scalar_f32) {
      g_l2_f32 = fn__l2_avx2_f32;
    }
    g_l2_i8 = fn__l2_avx2_i8;
  }
#endif

#ifdef FN_USE_SSE4
  if (fn_cpu_supports_sse4()) {
    if (g_l2_i8 == fn__l2_scalar_i8) {
      g_l2_i8 = fn__l2_sse4_i8;
    }
  }
#endif

#ifdef FN_USE_SSE
  if (fn_cpu_supports_sse()) {
    if (g_l2_f32 == fn__l2_scalar_f32) {
      g_l2_f32 = fn__l2_sse_f32;
    }
  }
#endif
}
#endif /* !FN_DPU_TARGET */

/*============================================================================
 * Distance Wrapper Functions
 *============================================================================*/

static float fn__l2_compute_f32(const void* x, const void* y, uint32_t dim, void* ctx) {
  (void)ctx;
  return g_l2_f32((const float*)x, (const float*)y, dim);
}

static float fn__l2_compute_i8(const void* x, const void* y, uint32_t dim, void* ctx) {
  (void)ctx;
  return g_l2_i8((const int8_t*)x, (const int8_t*)y, dim);
}

static float fn__l2_compute_u8(const void* x, const void* y, uint32_t dim, void* ctx) {
  (void)ctx;
  return g_l2_u8((const uint8_t*)x, (const uint8_t*)y, dim);
}

/* Transform functions (identity for L2) */
static void fn__l2_transform_f32(void* dest, const void* src, uint32_t dim, void* ctx) {
  (void)ctx;
  memcpy(dest, src, dim * sizeof(float));
}

static void fn__l2_transform_i8(void* dest, const void* src, uint32_t dim, void* ctx) {
  (void)ctx;
  memcpy(dest, src, dim * sizeof(int8_t));
}

static void fn__l2_transform_u8(void* dest, const void* src, uint32_t dim, void* ctx) {
  (void)ctx;
  memcpy(dest, src, dim * sizeof(uint8_t));
}

/*============================================================================
 * Public API
 *============================================================================*/

FN_API fn_error_t fn_distance_create_l2(fn_distance_t** out_distance, uint32_t dimension,
                                        fn_data_type_t data_type) {
  if (out_distance == NULL) {
    return FN_ERR_NULL_PTR;
  }

  if (dimension == 0) {
    return FN_ERR_INVALID_ARG;
  }

  fn_distance_t* dist = (fn_distance_t*)malloc(sizeof(fn_distance_t));
  if (dist == NULL) {
    return FN_ERR_OUT_OF_MEMORY;
  }

  memset(dist, 0, sizeof(fn_distance_t));
  dist->metric = FN_METRIC_L2;
  dist->data_type = data_type;
  dist->dimension = dimension;
  dist->context = NULL;

  switch (data_type) {
    case FN_DATA_FLOAT32:
      dist->ops.compute = fn__l2_compute_f32;
      dist->ops.compute_asym = NULL;
      dist->ops.transform = fn__l2_transform_f32;
      dist->data_size_bytes = dimension * sizeof(float);
      break;

    case FN_DATA_INT8:
      dist->ops.compute = fn__l2_compute_i8;
      dist->ops.compute_asym = NULL;
      dist->ops.transform = fn__l2_transform_i8;
      dist->data_size_bytes = dimension * sizeof(int8_t);
      break;

    case FN_DATA_UINT8:
      dist->ops.compute = fn__l2_compute_u8;
      dist->ops.compute_asym = NULL;
      dist->ops.transform = fn__l2_transform_u8;
      dist->data_size_bytes = dimension * sizeof(uint8_t);
      break;

    default:
      free(dist);
      return FN_ERR_NOT_SUPPORTED;
  }

  dist->ops.destroy = NULL;

  *out_distance = dist;
  return FN_ERR_OK;
}

FN_API void fn_distance_destroy(fn_distance_t* distance) {
  if (distance == NULL) {
    return;
  }

  if (distance->ops.destroy && distance->context) {
    distance->ops.destroy(distance->context);
  }

  free(distance);
}
