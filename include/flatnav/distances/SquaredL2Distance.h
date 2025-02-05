#pragma once
#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/L2DistanceDispatcher.h>
#include <flatnav/util/Datatype.h>
#include <flatnav/util/SquaredL2SimdExtensions.h>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cstddef>  // for size_t
#include <cstring>  // for memcpy
#include <functional>
#include <iostream>
#include <type_traits>

// This is the base distance function implementation for the L2 distance on
// floating-point inputs. We provide specializations that use SIMD when
// supported by the compiler and compatible with the input _dimension.

namespace flatnav::distances {

using util::DataType;
using util::type_for_data_type;

template <DataType data_type = DataType::float32>
class SquaredL2Distance : public DistanceInterface<SquaredL2Distance<data_type>> {

  friend class DistanceInterface<SquaredL2Distance>;
  enum { DISTANCE_ID = 0 };

 public:
  SquaredL2Distance() = default;
  SquaredL2Distance(size_t dim) : _dimension(dim), _data_size_bytes(dim * util::size(data_type)) {}

  static std::unique_ptr<SquaredL2Distance<data_type>> create(size_t dim) {
    return std::make_unique<SquaredL2Distance<data_type>>(dim);
  }

  inline constexpr size_t getDimension() const { return _dimension; }

  constexpr float distanceImpl(const void* x, const void* y, [[maybe_unused]] bool asymmetric = false) const {
    return L2DistanceDispatcher::dispatch(static_cast<const typename type_for_data_type<data_type>::type*>(x),
                                          static_cast<const typename type_for_data_type<data_type>::type*>(y),
                                          _dimension);
  }

  inline DataType getDataTypeImpl() const { return data_type; }

 private:
  size_t _dimension;
  size_t _data_size_bytes;

  friend class ::cereal::access;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(_dimension, _data_size_bytes);
  }

  inline size_t dataSizeImpl() { return _data_size_bytes; }

  inline void transformDataImpl(void* destination, const void* src) {
    std::memcpy(destination, src, _data_size_bytes);
  }

  void getSummaryImpl() {
    std::cout << "\nSquaredL2Distance Parameters" << std::flush;
    std::cout << "\n-----------------------------"
              << "\n"
              << std::flush;
    std::cout << "Dimension: " << _dimension << "\n" << std::flush;
  }
};

}  // namespace flatnav::distances
