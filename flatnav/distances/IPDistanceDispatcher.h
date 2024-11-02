#pragma once

#include <flatnav/util/Datatype.h>
#include <flatnav/util/InnerProductSimdExtensions.h>
#include <flatnav/util/Macros.h>

namespace flatnav::distances {

template <typename T>
static float defaultInnerProduct(const T* x, const T* y, const size_t& dimension) {
  float inner_product = 0;
  for (size_t i = 0; i < dimension; i++) {
    inner_product += x[i] * y[i];
  }
  return 1.0f - inner_product;
}

template <typename T>
struct InnerProductImpl {
  static float computeDistance(const T* x, const T* y, const size_t& dimension) {
    return defaultInnerProduct<T>(x, y, dimension);
  }
};

template <>
struct InnerProductImpl<float> {
  static float computeDistance(const float* x, const float* y, const size_t& dimension) {
#if defined(USE_AVX512)
    if (platformSupportsAvx512()) {
      if (dimension % 16 == 0) {
        return util::computeIP_Avx512(x, y, dimension);
      }
      if (dimension % 4 == 0) {
#if defined(USE_AVX)
        return util::computeIP_Avx_4aligned(x, y, dimension);
#else
        return util::computeIP_Sse4Aligned(x, y, dimension);
#endif
      } else if (dimension > 16) {
        return util::computeIP_SseWithResidual_16(x, y, dimension);
      } else if (dimension > 4) {
        return util::computeIP_SseWithResidual_4(x, y, dimension);
      }
    }
#endif

#if defined(USE_AVX)
    if (platformSupportsAvx()) {
      if (dimension % 16 == 0) {
        return util::computeIP_Avx(x, y, dimension);
      }
      if (dimension % 4 == 0) {
        return util::computeIP_Avx_4aligned(x, y, dimension);
      } else if (dimension > 16) {
        return util::computeIP_SseWithResidual_16(x, y, dimension);
      } else if (dimension > 4) {
        return util::computeIP_SseWithResidual_4(x, y, dimension);
      }
    }
#endif

#if defined(USE_SSE)
    if (dimension % 16 == 0) {
      return util::computeIP_Sse(x, y, dimension);
    }
    if (dimension % 4 == 0) {
      return util::computeIP_Sse_4aligned(x, y, dimension);
    } else if (dimension > 16) {
      return util::computeIP_SseWithResidual_16(x, y, dimension);
    } else if (dimension > 4) {
      return util::computeIP_SseWithResidual_4(x, y, dimension);
    }

#endif
    return defaultInnerProduct<float>(x, y, dimension);
  }
};

// TODO: Include SIMD optimized implementations for int8_t.
template <>
struct InnerProductImpl<int8_t> {
  static float computeDistance(const int8_t* x, const int8_t* y, const size_t& dimension) {
    return defaultInnerProduct<int8_t>(x, y, dimension);
  }
};

// TODO: Include SIMD optimized implementations for uint8_t.
template <>
struct InnerProductImpl<uint8_t> {
  static float computeDistance(const uint8_t* x, const uint8_t* y, const size_t& dimension) {
    return defaultInnerProduct<uint8_t>(x, y, dimension);
  }
};

struct IPDistanceDispatcher {
  template <typename T>
  static float dispatch(const T* x, const T* y, const size_t& dimension) {
    return InnerProductImpl<T>::computeDistance(x, y, dimension);
  }
};

}  // namespace flatnav::distances