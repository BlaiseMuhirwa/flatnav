#pragma once

#include "../util/verifysimd.h"
#include "InnerProductDistance.h"

#include <cstddef> // for size_t

namespace flatnav {

// TODO: Check whether the use of the "exact dimension match" distances
// is actually any faster than the general sized ones.
#if defined(USE_AVX512)
class InnerProductDistance_SIMD16AVX512 : public InnerProductDistance {
  // Specialization for processing size-16 chunks with AVX512.
  float distance_impl(const void *x, const void *y, size_t &dimension) {
    return 1.0 - InnerProductSIMD16ResAVX512(x, y, dimension);
  }
}
#endif

#if defined(USE_AVX)
class InnerProductDistance_SIMD16AVX : public InnerProductDistance {
  // Specialization for processing size-16 chunks with AVX.
  float distance_impl(const void *x, const void *y, size_t &dimension) {
    return 1.0 - InnerProductSIMD16ResAVX(x, y, dimension);
  }
}
#endif

#if defined(USE_SSE)

static float InnerProductSIMD16ExtSSE(const void *x, const void *y,
                                      size_t &dimension) {}

static float InnerProductSIMD16ResSSE(const void *x, const void *y,
                                      size_t &dimension) {}

static float InnerProductSIMD4ExtSSE(const void *x, const void *y,
                                     size_t &dimension) {}

static float InnerProductSIMD4ResSSE(const void *x, const void *y,
                                     size_t &dimension) {}
class InnerProductDistance_SIMD16SSE : public InnerProductDistance {
  // Specialization for processing size-16 chunks with SSE.
  float distance_impl(const void *x, const void *y, size_t &dimension) {
    return 1.0 - InnerProductSIMD16ResSSE(x, y, dimension);
  }
};

class InnerProductDistance_SIMD4SSE : public InnerProductDistance {
  // Specialization for processing size-4 chunks with SSE.
  float distance_impl(const void *x, const void *y, size_t &dimension) {
    return 1.0 - InnerProductSIMD16ResSSE(x, y, dimension);
  }
};
#endif

// The rest of the file contains specialized distance function implementations.
// These are heavily inspired by the ones from hnswlib, but with some
// refactoring. They are optimized for speed rather than readability.

static float
InnerProduct(const void *x, const void *y, size_t &dimension) {
}

#if defined(USE_AVX512)
static float InnerProductSIMD16ExtAVX512(const void *x, const void *y,
                                         size_t &dimension) {}

static float InnerProductSIMD16ResAVX512(const void *x, const void *y,
                                         size_t &dimension) {}
#endif

#if defined(USE_AVX)
static float InnerProductSIMD16ExtAVX(const void *x, const void *y,
                                      size_t &dimension) {}

static float InnerProductSIMD16ResAVX(const void *x, const void *y,
                                      size_t &dimension) {}
#endif

#if defined(USE_SSE)

#endif

} // namespace flatnav