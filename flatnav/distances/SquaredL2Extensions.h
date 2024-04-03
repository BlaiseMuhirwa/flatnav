#pragma once

#include <flatnav/DistanceInterface.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/util/Macros.h>
#include <flatnav/util/SimdAvx.h>

using flatnav::util::simd16float32;
using flatnav::util::simd8float32;

namespace flatnav {

class SquaredL2Avx512 : public SquaredL2Distance {
  //   friend class DistanceInterface<SquaredL2Avx512>;

public:
  explicit SquaredL2Avx512(size_t dim) : SquaredL2Distance(dim) {
    std::cout << "Using AVX512" << std::endl;
  }

  float distanceImpl(const void *x, const void *y,
                     bool asymmetric = false) const {
    (void)asymmetric;

    std::cout << "AVX512" << std::endl;

    float *pointer_x = (float *)x;
    float *pointer_y = (float *)y;
    const float *end_x = pointer_x + (_dimension >> 4 << 4);
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
};

} // namespace flatnav