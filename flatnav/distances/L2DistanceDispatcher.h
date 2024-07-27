#pragma once

#include <flatnav/util/Macros.h>
#include <flatnav/util/SquaredL2SimdExtensions.h>

namespace flatnav::distances {

template <size_t dimension, typename T>
static float defaultSquaredL2(const T *x, const T *y) {
  float squared_distance = 0;
  for (size_t i = 0; i < dimension; i++) {
    float difference = x[i] - y[i];
    squared_distance += difference * difference;
  }
  return squared_distance;
}

template <typename T> struct SquaredL2Impl {
  static float computeDistance(const void *x, const void *y,
                               const size_t &dimension) {
    return defaultSquaredL2<T>(static_cast<const T *>(x),
                               static_cast<const T *>(y));
  }
}

template <>
struct SquaredL2Impl<float> {
  static float computeDistance(const void *x, const void *y,
                               const size_t &dimension) {
#if defined(USE_AVX512)
    if (platformSupportsAvx512()) {
      return util::computeL2_Avx512(static_cast<const float *>(x),
                                    static_cast<const float *>(y), dimension);
    }
#endif

#if defined(USE_AVX)
    if (platformSupportsAvx()) {
      return util::computeL2_Avx2(static_cast<const float *>(x),
                                  static_cast<const float *>(y), dimension);
    }
#endif

#if defined(USE_SSE)
    return util::computeL2_Sse(static_cast<const float *>(x),
                               static_cast<const float *>(y), dimension);
#endif

    return defaultSquaredL2<dimension, float>(static_cast<const float *>(x),
                                              static_cast<const float *>(y));
  }
};

template<>
struct SquaredL2Impl<int8_t> {
  static float computeDistance(const void *x, const void *y,
                               const size_t &dimension) {
    
    }
} // namespace flatnav::distances