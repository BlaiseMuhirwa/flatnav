#pragma once
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cstddef> // for size_t
#include <cstring> // for memcpy
#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/L2DistanceDispatcher.h>
#include <flatnav/util/Datatype.h>
#include <flatnav/util/SquaredL2SimdExtensions.h>
#include <functional>
#include <iostream>
#include <type_traits>

// This is the base distance function implementation for the L2 distance on
// floating-point inputs. We provide specializations that use SIMD when
// supported by the compiler and compatible with the input _dimension.

namespace flatnav::distances {

using util::DataType;
using util::type_for_data_type;

template <DataType data_type = DataType::float32> struct OptimalL2SimdSelector {

  static void
  adjustForNonOptimalDimensions(DistanceFunctionPtr &distance_function,
                                const size_t &dimension) {
#if defined(USE_SSE)
    if (dimension % 16 != 0) {
      if (dimension % 4 == 0) {
        distance_function = std::make_unique<DistanceFunction>(
            std::bind(&util::computeL2_Sse4Aligned, std::placeholders::_1,
                      std::placeholders::_2, std::cref(dimension)));

      } else if (dimension > 16) {
        distance_function = std::make_unique<DistanceFunction>(std::bind(
            &util::computeL2_SseWithResidual_16, std::placeholders::_1,
            std::placeholders::_2, std::cref(dimension)));

      } else if (dimension > 4) {
        distance_function = std::make_unique<DistanceFunction>(
            std::bind(&util::computeL2_SseWithResidual_4, std::placeholders::_1,
                      std::placeholders::_2, std::cref(dimension)));
      }
    }
#endif // USE_SSE
  }

  static void selectInt8(DistanceFunctionPtr &distance_function,
                         const size_t &dimension) {
#if defined(USE_AVX512)
    if (platformSupportsAvx512()) {
      distance_function = std::make_unique<DistanceFunction>(
          std::bind(&util::computeL2_Sse_int8, std::placeholders::_1,
                    std::placeholders::_2, std::cref(dimension)));
    }
#endif // USE_AVX512
  }

  static void selectUint8(DistanceFunctionPtr &distance_function,
                          const size_t &dimension) {
#if defined(USE_AVX512)
    distance_function = std::make_unique<DistanceFunction>(
        std::bind(&util::computeL2_Avx512_Uint8, std::placeholders::_1,
                  std::placeholders::_2, std::cref(dimension)));
#endif // USE_AVX512
  }

  static void selectFloat32(DistanceFunctionPtr &distance_function,
                            const size_t &dimension) {
#if defined(USE_SSE)
    distance_function = std::make_unique<DistanceFunction>(
        std::bind(&util::computeL2_Sse, std::placeholders::_1,
                  std::placeholders::_2, std::cref(dimension)));
#endif // USE_SSE

#if defined(USE_AVX512)
    if (platformSupportsAvx512()) {
      distance_function = std::make_unique<DistanceFunction>(
          std::bind(&util::computeL2_Avx512, std::placeholders::_1,
                    std::placeholders::_2, std::cref(dimension)));

      adjustForNonOptimalDimensions(distance_function, dimension);
      return;
    }
#endif // USE_AVX512

#if defined(USE_AVX)
    if (platformSupportsAvx()) {

      distance_function = std::make_unique<DistanceFunction>(
          std::bind(&util::computeL2_Avx2, std::placeholders::_1,
                    std::placeholders::_2, std::cref(dimension)));
      adjustForNonOptimalDimensions(distance_function, dimension);
      return;
    }
#endif // USE_AVX

    adjustForNonOptimalDimensions(distance_function, dimension);
  }

  /**
   * @brief Select the optimal distance function based on the input dimension
   * @param dimension The dimension of the input data
   *
   * @note There are different SIMD functions for float32, int8_t and uint8_t.
   * This is why we are templating this class on the data type.
   */
  static void select(DistanceFunctionPtr &distance_function,
                     const size_t &dimension) {
    switch (data_type) {
    case DataType::float32:
      selectFloat32(distance_function, dimension);
      break;
    case DataType::int8:
      selectInt8(distance_function, dimension);
      break;
    case DataType::uint8:
      selectUint8(distance_function, dimension);
      break;
    default:
      throw std::runtime_error("Unsupported data type");
    }
  }
};

struct DefaultSquaredL2 {
  template <typename T>
  static constexpr float compute(const void *x, const void *y,
                                 const size_t &dimension) {
    T *p_x = const_cast<T *>(static_cast<const T *>(x));
    T *p_y = const_cast<T *>(static_cast<const T *>(y));
    float squared_distance = 0;
    for (size_t i = 0; i < dimension; i++) {
      float difference = *p_x - *p_y;
      p_x++;
      p_y++;
      squared_distance += difference * difference;
    }
    return squared_distance;
  }
};


/**
 * @brief The SquaredL2Distance class is designed to balance compile-time and
 * runtime dispatching for efficient distance computation.
 *
 * This class template takes a util::DataType parameter, which allows it to
 * leverage compile-time dispatch for selecting the appropriate distance
 * computation strategy based on the data type. The compile-time dispatch is
 * achieved using template specialization and the TypeForDataType struct, which
 * maps util::DataType values to corresponding C++ types.
 *
 * The actual distance computation is performed by the distance::DistanceL2
 * class, which relies on the L2Impl struct and its specializations. The
 * appropriate L2Impl specialization is selected at compile-time based on the
 * data type provided.
 *
 * For SIMD operations, which require runtime checks to determine hardware
 * support (e.g., AVX, AVX512), the distance::L2Impl specializations include
 * runtime checks to select the appropriate SIMD implementation. These runtime
 * checks ensure that the most efficient SIMD instructions available on the
 * platform are used.
 *
 * The key points of this design are:
 * 1. Compile-time dispatch is used to select the appropriate distance
 * computation strategy based on the data type, reducing runtime overhead and
 * improving performance.
 * 2. Runtime checks are used to determine the availability of SIMD
 * instructions, ensuring that the best possible SIMD implementation is used
 * based on the hardware capabilities.
 * 3. The distanceImpl method in SquaredL2Distance uses the TypeForDataType
 * struct to determine the C++ type corresponding to the util::DataType at
 * compile-time, ensuring the correct L2Impl specialization is used.
 *
 * This design provides a balance between compile-time efficiency and runtime
 * flexibility, making it well-suited for performance-critical paths where
 * different data types and hardware capabilities must be handled efficiently.
 */

template <DataType data_type>
class SquaredL2Distance
    : public DistanceInterface<SquaredL2Distance<data_type>> {

  friend class DistanceInterface<SquaredL2Distance>;
  enum { DISTANCE_ID = 0 };

public:
  SquaredL2Distance() = default;
  SquaredL2Distance(size_t dim)
      : _dimension(dim), _data_size_bytes(dim * util::size(data_type)) {}

  static std::unique_ptr<SquaredL2Distance<data_type>> create(size_t dim) {
    return std::make_unique<SquaredL2Distance<data_type>>(dim);
  }

  inline constexpr size_t getDimension() const { return _dimension; }

  constexpr float distanceImpl(const void *x, const void *y,
                               bool asymmetric = false) const {
    (void)asymmetric;
    return L2DistanceDispatcher::dispatch(
        static_cast<const typename type_for_data_type<data_type>::type *>(x),
        static_cast<const typename type_for_data_type<data_type>::type *>(y),
        _dimension);
  }

private:
  size_t _dimension;
  size_t _data_size_bytes;

  friend class ::cereal::access;

  template <typename Archive> void serialize(Archive &ar) {
    ar(_dimension, _data_size_bytes);
  }

  inline size_t dataSizeImpl() { return _data_size_bytes; }

  inline void transformDataImpl(void *destination, const void *src) {
    std::memcpy(destination, src, _data_size_bytes);
  }

  void getSummaryImpl() {
    std::cout << "\nSquaredL2Distance Parameters" << std::flush;
    std::cout << "\n-----------------------------"
              << "\n"
              << std::flush;
    std::cout << "Dimension: " << _dimension << "\n" << std::flush;
  }

  float defaultDistanceImpl(const void *x, const void *y,
                            const size_t &dimension) const {
    if (data_type == DataType::float32) {
      return DefaultSquaredL2::compute<float>(x, y, dimension);
    } else if (data_type == DataType::int8) {
      return DefaultSquaredL2::compute<int8_t>(x, y, dimension);
    } else if (data_type == DataType::uint8) {
      return DefaultSquaredL2::compute<uint8_t>(x, y, dimension);
    }
    throw std::runtime_error("Unsupported data type");
  }
};

} // namespace flatnav::distances
