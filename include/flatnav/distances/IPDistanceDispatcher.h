#pragma once

#include <flatnav/util/Datatype.h>
#include <flatnav/util/InnerProductSimdExtensions.h>
#include <flatnav/util/Macros.h>

namespace flatnav::distances {

template <typename T>
static float default_inner_product(const T* x, const T* y, const size_t& dimension) {
  float inner_product = 0;
  for (size_t i = 0; i < dimension; i++) {
    inner_product += x[i] * y[i];
  }
  return 1.0f - inner_product;
}

template <typename T>
struct InnerProductImpl {
  static float computeDistance(const T* x, const T* y, const size_t& dimension) {
    return default_inner_product<T>(x, y, dimension);
  }
};

template <>
struct InnerProductImpl<float> {
  static float computeDistance(const float* x, const float* y, const size_t& dimension) {
#if defined(USE_AVX512)
    if (platformSupportsAvx512()) {
      if (dimension % 16 == 0) {
        return util::compute_ip_avx512(x, y, dimension);
      }
      if (dimension % 4 == 0) {
#if defined(USE_AVX)
        return util::compute_ip_avx_4aligned(x, y, dimension);
#else
        return util::compute_ip_sse_4aligned(x, y, dimension);
#endif
      } else if (dimension > 16) {
        return util::compute_ip_sse_residual_16(x, y, dimension);
      } else if (dimension > 4) {
        return util::compute_ip_sse_residual_4(x, y, dimension);
      }
    }
#endif

#if defined(USE_AVX)
    if (platformSupportsAvx()) {
      if (dimension % 16 == 0) {
        return util::compute_ip_avx(x, y, dimension);
      }
      if (dimension % 4 == 0) {
        return util::compute_ip_avx_4aligned(x, y, dimension);
      } else if (dimension > 16) {
        return util::compute_ip_sse_residual_16(x, y, dimension);
      } else if (dimension > 4) {
        return util::compute_ip_sse_residual_4(x, y, dimension);
      }
    }
#endif

#if defined(USE_SSE)
    if (dimension % 16 == 0) {
      return util::compute_ip_sse(x, y, dimension);
    }
    if (dimension % 4 == 0) {
      return util::compute_ip_sse_4aligned(x, y, dimension);
    } else if (dimension > 16) {
      return util::compute_ip_sse_residual_16(x, y, dimension);
    } else if (dimension > 4) {
      return util::compute_ip_sse_residual_4(x, y, dimension);
    }

#endif
    return default_inner_product<float>(x, y, dimension);
  }
};

template <>
struct InnerProductImpl<int8_t> {
  static float computeDistance(const int8_t* x, const int8_t* y, const size_t& dimension) {
#if defined(USE_AVX512)
    if (platformSupportsAvx512()) {
      return util::compute_ip_avx512_int8(x, y, dimension);
    }
#endif

#if defined(USE_AVX)
    if (platformSupportsAvx()) {
      return util::compute_ip_avx2_int8(x, y, dimension);
    }
#endif

#if defined(USE_SSE4_1)
    return util::compute_ip_sse_int8(x, y, dimension);
#endif

    return default_inner_product<int8_t>(x, y, dimension);
  }
};

template <>
struct InnerProductImpl<uint8_t> {
  static float computeDistance(const uint8_t* x, const uint8_t* y, const size_t& dimension) {
#if defined(USE_AVX512)
    if (platformSupportsAvx512()) {
      return util::compute_ip_avx512_uint8(x, y, dimension);
    }
#endif

#if defined(USE_AVX)
    if (platformSupportsAvx()) {
      return util::compute_ip_avx2_uint8(x, y, dimension);
    }
#endif

#if defined(USE_SSE4_1)
    return util::compute_ip_sse_uint8(x, y, dimension);
#endif

    return default_inner_product<uint8_t>(x, y, dimension);
  }
};

struct IPDistanceDispatcher {
  template <typename T>
  static float dispatch(const T* x, const T* y, const size_t& dimension) {
    return InnerProductImpl<T>::computeDistance(x, y, dimension);
  }
};

}  // namespace flatnav::distances
