/**
 * @file distance_ip.c
 * @brief Inner product distance implementations
 *
 * Inner product distance is computed as (1 - dot_product) for normalized
 * vectors, so that lower values indicate more similar vectors.
 */

#include <flatnav/flatnav.h>
#include <stdlib.h>
#include <string.h>

/*============================================================================
 * Scalar Implementations (Always Available)
 *============================================================================*/

static float fn__ip_scalar_f32(const float* x, const float* y, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; i++) {
    sum += x[i] * y[i];
  }
  /* Return 1 - dot_product so lower = more similar */
  return 1.0f - sum;
}

static float fn__ip_scalar_i8(const int8_t* x, const int8_t* y, uint32_t dim) {
  int32_t sum = 0;
  for (uint32_t i = 0; i < dim; i++) {
    sum += (int32_t)x[i] * (int32_t)y[i];
  }
  /* Scale factor: assuming int8 normalized to [-127, 127] range */
  /* Max dot product would be dim * 127 * 127 */
  return 1.0f - (float)sum / (float)(dim * 127 * 127);
}

static float fn__ip_scalar_u8(const uint8_t* x, const uint8_t* y, uint32_t dim) {
  uint32_t sum = 0;
  for (uint32_t i = 0; i < dim; i++) {
    sum += (uint32_t)x[i] * (uint32_t)y[i];
  }
  /* Scale factor: assuming uint8 normalized to [0, 255] range */
  return 1.0f - (float)sum / (float)(dim * 255 * 255);
}

/*============================================================================
 * SSE Implementations
 *============================================================================*/

#ifdef FN_USE_SSE
#include <emmintrin.h>
#include <xmmintrin.h>

__attribute__((target("sse"))) static float fn__ip_sse_f32(const float* x, const float* y, uint32_t dim) {
  __m128 sum = _mm_setzero_ps();
  uint32_t aligned_dim = dim & ~3u;

  for (uint32_t i = 0; i < aligned_dim; i += 4) {
    __m128 vx = _mm_loadu_ps(x + i);
    __m128 vy = _mm_loadu_ps(y + i);
    sum = _mm_add_ps(sum, _mm_mul_ps(vx, vy));
  }

  /* Horizontal sum */
  __m128 shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
  sum = _mm_add_ps(sum, shuf);
  shuf = _mm_movehl_ps(shuf, sum);
  sum = _mm_add_ss(sum, shuf);

  float result = _mm_cvtss_f32(sum);

  /* Handle residual */
  for (uint32_t i = aligned_dim; i < dim; i++) {
    result += x[i] * y[i];
  }

  return 1.0f - result;
}

#endif /* FN_USE_SSE */

/*============================================================================
 * SSE4.1 Implementations
 *============================================================================*/

#ifdef FN_USE_SSE4
#include <smmintrin.h>

__attribute__((target("sse4.1"))) static float fn__ip_sse4_i8(const int8_t* x, const int8_t* y,
                                                              uint32_t dim) {
  __m128i sum = _mm_setzero_si128();
  uint32_t aligned_dim = dim & ~15u;

  for (uint32_t i = 0; i < aligned_dim; i += 16) {
    __m128i vx = _mm_loadu_si128((const __m128i*)(x + i));
    __m128i vy = _mm_loadu_si128((const __m128i*)(y + i));

    /* Unpack to 16-bit for multiplication */
    __m128i x_lo = _mm_cvtepi8_epi16(vx);
    __m128i x_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vx, 8));
    __m128i y_lo = _mm_cvtepi8_epi16(vy);
    __m128i y_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vy, 8));

    /* Multiply and horizontal add pairs */
    __m128i prod_lo = _mm_madd_epi16(x_lo, y_lo);
    __m128i prod_hi = _mm_madd_epi16(x_hi, y_hi);

    sum = _mm_add_epi32(sum, prod_lo);
    sum = _mm_add_epi32(sum, prod_hi);
  }

  /* Horizontal sum */
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  int32_t result = _mm_cvtsi128_si32(sum);

  /* Handle residual */
  for (uint32_t i = aligned_dim; i < dim; i++) {
    result += (int32_t)x[i] * (int32_t)y[i];
  }

  return 1.0f - (float)result / (float)(dim * 127 * 127);
}

#endif /* FN_USE_SSE4 */

/*============================================================================
 * AVX2 Implementations
 *============================================================================*/

#ifdef FN_USE_AVX2
#include <immintrin.h>

__attribute__((target("avx2,fma"))) static float fn__ip_avx2_f32(const float* x, const float* y,
                                                                 uint32_t dim) {
  __m256 sum = _mm256_setzero_ps();
  uint32_t aligned_dim = dim & ~7u;

  for (uint32_t i = 0; i < aligned_dim; i += 8) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = _mm256_loadu_ps(y + i);
    sum = _mm256_fmadd_ps(vx, vy, sum);
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
    result += x[i] * y[i];
  }

  return 1.0f - result;
}

__attribute__((target("avx2"))) static float fn__ip_avx2_i8(const int8_t* x, const int8_t* y, uint32_t dim) {
  __m256i sum = _mm256_setzero_si256();
  uint32_t aligned_dim = dim & ~31u;

  for (uint32_t i = 0; i < aligned_dim; i += 32) {
    __m256i vx = _mm256_loadu_si256((const __m256i*)(x + i));
    __m256i vy = _mm256_loadu_si256((const __m256i*)(y + i));

    /* Sign-extend to 16-bit */
    __m256i x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vx));
    __m256i x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx, 1));
    __m256i y_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vy));
    __m256i y_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vy, 1));

    /* Multiply and add pairs to 32-bit */
    __m256i prod_lo = _mm256_madd_epi16(x_lo, y_lo);
    __m256i prod_hi = _mm256_madd_epi16(x_hi, y_hi);

    sum = _mm256_add_epi32(sum, prod_lo);
    sum = _mm256_add_epi32(sum, prod_hi);
  }

  /* Horizontal sum */
  __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
  sum128 = _mm_hadd_epi32(sum128, sum128);
  sum128 = _mm_hadd_epi32(sum128, sum128);
  int32_t result = _mm_cvtsi128_si32(sum128);

  /* Handle residual */
  for (uint32_t i = aligned_dim; i < dim; i++) {
    result += (int32_t)x[i] * (int32_t)y[i];
  }

  return 1.0f - (float)result / (float)(dim * 127 * 127);
}

#endif /* FN_USE_AVX2 */

/*============================================================================
 * AVX-512 Implementations
 *============================================================================*/

#ifdef FN_USE_AVX512
#include <immintrin.h>

__attribute__((target("avx512f"))) static float fn__ip_avx512_f32(const float* x, const float* y,
                                                                  uint32_t dim) {
  __m512 sum = _mm512_setzero_ps();
  uint32_t aligned_dim = dim & ~15u;

  for (uint32_t i = 0; i < aligned_dim; i += 16) {
    __m512 vx = _mm512_loadu_ps(x + i);
    __m512 vy = _mm512_loadu_ps(y + i);
    sum = _mm512_fmadd_ps(vx, vy, sum);
  }

  float result = _mm512_reduce_add_ps(sum);

  /* Handle residual */
  for (uint32_t i = aligned_dim; i < dim; i++) {
    result += x[i] * y[i];
  }

  return 1.0f - result;
}

#endif /* FN_USE_AVX512 */

/*============================================================================
 * Distance Function Dispatch
 *============================================================================*/

/* Function pointer types */
typedef float (*fn__ip_f32_func_t)(const float*, const float*, uint32_t);
typedef float (*fn__ip_i8_func_t)(const int8_t*, const int8_t*, uint32_t);
typedef float (*fn__ip_u8_func_t)(const uint8_t*, const uint8_t*, uint32_t);

/* Global dispatch table */
static fn__ip_f32_func_t g_ip_f32 = fn__ip_scalar_f32;
static fn__ip_i8_func_t g_ip_i8 = fn__ip_scalar_i8;
static fn__ip_u8_func_t g_ip_u8 = fn__ip_scalar_u8;

#ifndef FN_DPU_TARGET
void fn__init_ip_dispatch(void) {
  /* Select best available implementation */
#ifdef FN_USE_AVX512
  if (fn_cpu_supports_avx512()) {
    g_ip_f32 = fn__ip_avx512_f32;
  }
#endif

#ifdef FN_USE_AVX2
  if (fn_cpu_supports_avx2()) {
    if (g_ip_f32 == fn__ip_scalar_f32) {
      g_ip_f32 = fn__ip_avx2_f32;
    }
    g_ip_i8 = fn__ip_avx2_i8;
  }
#endif

#ifdef FN_USE_SSE4
  if (fn_cpu_supports_sse4()) {
    if (g_ip_i8 == fn__ip_scalar_i8) {
      g_ip_i8 = fn__ip_sse4_i8;
    }
  }
#endif

#ifdef FN_USE_SSE
  if (fn_cpu_supports_sse()) {
    if (g_ip_f32 == fn__ip_scalar_f32) {
      g_ip_f32 = fn__ip_sse_f32;
    }
  }
#endif
}
#endif /* !FN_DPU_TARGET */

/*============================================================================
 * Distance Wrapper Functions
 *============================================================================*/

static float fn__ip_compute_f32(const void* x, const void* y, uint32_t dim, void* ctx) {
  (void)ctx;
  return g_ip_f32((const float*)x, (const float*)y, dim);
}

static float fn__ip_compute_i8(const void* x, const void* y, uint32_t dim, void* ctx) {
  (void)ctx;
  return g_ip_i8((const int8_t*)x, (const int8_t*)y, dim);
}

static float fn__ip_compute_u8(const void* x, const void* y, uint32_t dim, void* ctx) {
  (void)ctx;
  return g_ip_u8((const uint8_t*)x, (const uint8_t*)y, dim);
}

/* Transform functions (identity for IP) */
static void fn__ip_transform_f32(void* dest, const void* src, uint32_t dim, void* ctx) {
  (void)ctx;
  memcpy(dest, src, dim * sizeof(float));
}

static void fn__ip_transform_i8(void* dest, const void* src, uint32_t dim, void* ctx) {
  (void)ctx;
  memcpy(dest, src, dim * sizeof(int8_t));
}

static void fn__ip_transform_u8(void* dest, const void* src, uint32_t dim, void* ctx) {
  (void)ctx;
  memcpy(dest, src, dim * sizeof(uint8_t));
}

/*============================================================================
 * Public API
 *============================================================================*/

FN_API fn_error_t fn_distance_create_ip(fn_distance_t** out_distance, uint32_t dimension,
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
  dist->metric = FN_METRIC_IP;
  dist->data_type = data_type;
  dist->dimension = dimension;
  dist->context = NULL;

  switch (data_type) {
    case FN_DATA_FLOAT32:
      dist->ops.compute = fn__ip_compute_f32;
      dist->ops.compute_asym = NULL;
      dist->ops.transform = fn__ip_transform_f32;
      dist->data_size_bytes = dimension * sizeof(float);
      break;

    case FN_DATA_INT8:
      dist->ops.compute = fn__ip_compute_i8;
      dist->ops.compute_asym = NULL;
      dist->ops.transform = fn__ip_transform_i8;
      dist->data_size_bytes = dimension * sizeof(int8_t);
      break;

    case FN_DATA_UINT8:
      dist->ops.compute = fn__ip_compute_u8;
      dist->ops.compute_asym = NULL;
      dist->ops.transform = fn__ip_transform_u8;
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
