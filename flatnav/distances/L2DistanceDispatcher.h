#pragma once

#include <flatnav/util/Datatype.h>
#include <flatnav/util/Macros.h>
#include <flatnav/util/SquaredL2SimdExtensions.h>

namespace flatnav::distances {

template <typename T>
static float defaultSquaredL2(const T *x, const T *y, const size_t &dimension) {
  float squared_distance = 0;
  for (size_t i = 0; i < dimension; i++) {
    float difference = x[i] - y[i];
    squared_distance += difference * difference;
  }
  return squared_distance;
}

// This struct provides a generic implementation of computing the squared L2
// distance
//  between two arrays of type T.
// @TODO: We should add constraints to the T type.
template <typename T> struct SquaredL2Impl {
  /**
   * Computes the squared L2 distance between two arrays of type T.
   *
   * @param x The first array.
   * @param y The second array.
   * @param dimension The dimension of the arrays.
   * @return The squared L2 distance between the two arrays.
   */
  static float computeDistance(const T *x, const T *y,
                               const size_t &dimension) {
    return defaultSquaredL2<T>(x, y, dimension);
  }
};

// Specialization of SquaredL2Impl for the float type.
template <> struct SquaredL2Impl<float> {
  static float computeDistance(const float *x, const float *y,
                               const size_t &dimension) {
#if defined(USE_AVX512)
    if (platformSupportsAvx512()) {
      return util::computeL2_Avx512(x, y, dimension);
    }
#endif

#if defined(USE_AVX)
    if (platformSupportsAvx()) {
      return util::computeL2_Avx2(x, y, dimension);
    }
#endif

#if defined(USE_SSE)
    return util::computeL2_Sse(x, y, dimension);
#else
    return defaultSquaredL2<float>(x, y, dimension);
#endif
  }
};

template <> struct SquaredL2Impl<int8_t> {
  static float computeDistance(const int8_t *x, const int8_t *y,
                               const size_t &dimension) {
#if defined(USE_AVX512BW) && defined(USE_AVX512VNNI)
    if (platformSupportsAvx512()) {
      return flatnav::util::computeL2_Avx512_int8(x, y, dimension);
    }
#endif
#if defined(USE_SSE)
    return flatnav::util::computeL2_Sse_int8(x, y, dimension);
#endif
    return defaultSquaredL2<int8_t>(x, y, dimension);
  }
};

template <> struct SquaredL2Impl<uint8_t> {
  static float computeDistance(const uint8_t *x, const uint8_t *y,
                               const size_t &dimension) {
#if defined(USE_AVX512)
    if (platformSupportsAvx512()) {
      return util::computeL2_Avx512_Uint8(x, y, dimension);
    }
#endif

    return defaultSquaredL2<uint8_t>(x, y, dimension);
  }
};

struct L2DistanceDispatcher {
  template <typename T>
  static float dispatch(const T *x, const T *y, const size_t &dimension) {
    return SquaredL2Impl<T>::computeDistance(x, y, dimension);
  }
};

} // namespace flatnav::distances