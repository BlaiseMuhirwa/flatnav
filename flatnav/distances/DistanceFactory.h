#pragma once

#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/distances/SquaredL2Extensions.h>
#include <flatnav/util/Macros.h>

namespace flatnav {

struct SquaredL2DistanceFactory {

  static std::shared_ptr<SquaredL2Distance> create(size_t dim) {
    auto l2_implementer = selectL2Implementer(dim);
    return l2_implementer;
  }

  static void upgradeToAdvancedSimdIfAvailable(
      std::shared_ptr<SquaredL2Distance> &implementer) {
#if defined(USE_AVX512) || defined(USE_AVX)
    auto dim = implementer->getDimension();
    if (platformSupportsAvx512()) {
      implementer = std::make_shared<SquaredL2Avx512>(dim);
    } else if (platformSupportsAvx()) {
      implementer = std::make_shared<SquaredL2Avx2>(dim);
    }
#endif
  }

  static std::shared_ptr<SquaredL2Distance> selectL2Implementer(size_t dim) {
    auto implementer = std::make_shared<SquaredL2Distance>();
#ifndef NO_SIMD_VECTORIZATION
    // start with a default SSE implementation
    implementer = std::make_shared<SquaredL2DistanceSIMD4Ext>(dim);

    // Upgrade to more advanced SIMD if available
    upgradeToAdvancedSimdIfAvailable(implementer);

    // Adjust for specific dimensions
    adjustForDimensions(implementer);

#endif // NO_MANUAL_VECTORIZATION
  }

  static void adjustForDimensions(std::shared_ptr<SquaredL2Distance>& implementer) {
    auto dim = implementer->getDimension();
    if (dim % 16 != 0) {
      if (dim % 4 == 0) {
        implementer = std::make_shared<SquaredL2DistanceSIMD4Ext>(dim);
      } else if (dim > 16) {
        implementer = std::make_shared<SquaredL2DistanceSIMD16ExtResiduals>(dim);
      } else if (dim > 4) {
        implementer = std::make_shared<SquaredL2DistanceSIMD4ExtResiduals>(dim);
      }
    }
  }

};

} // namespace flatnav