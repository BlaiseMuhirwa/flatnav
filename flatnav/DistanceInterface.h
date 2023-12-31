#pragma once

#include <cereal/access.hpp>
#include <cstddef> // for size_t
#include <fstream> // for ifstream, ofstream
#include <iostream>

namespace flatnav {

enum class METRIC_TYPE { EUCLIDEAN, INNER_PRODUCT };
enum class ARCH_OPTIMIZATION { NONE, SSE, AVX, AVX512 };

// We use the CRTP to implement static polymorphism on the distance. This is
// done to allow for metrics and distance functions that support arbitrary
// pre-processing (such as quantization etc) without having to call the
// distance function through a pointer or virtual function call.
// CRTP: https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern

template <typename T> class DistanceInterface {
public:
  // The asymmetric flag is used to indicate whether the distance function
  // is between two database vectors (symmetric) or between a database vector
  // and a query vector. For regular distances (l2, inner product), there is
  // no difference between the two. However, for quantization techniques, such
  // as product quantization, the two distance modes are different.
  float distance(const void *x, const void *y, bool asymmetric = false) {
    return static_cast<T *>(this)->distanceImpl(x, y, asymmetric);
  }

  // Returns the dimension of the input data.
  size_t dimension() { return static_cast<T *>(this)->getDimension(); }

  // Returns the size, in bytes, of the transformed data representation.
  size_t dataSize() { return static_cast<T *>(this)->dataSizeImpl(); }

  // Prints the parameters of the distance function.
  void printParams() { static_cast<T *>(this)->printParamsImpl(); }

  // This transforms the data located at src into a form that is writeable
  // to disk / storable in RAM. For distance functions that don't
  // compress the input, this just passses through a copy from src to
  // destination. However, there are functions (e.g. with quantization) where
  // the in-memory representation is not the same as the raw input.
  void transformData(void *destination, const void *src) {
    static_cast<T *>(this)->transformDataImpl(destination, src);
  }

  // Serializes the distance function to disk.
  template <typename Archive> void serialize(Archive &archive) {
    static_cast<T *>(this)->template serialize<Archive>(archive);
  }
};

#define SELECT_DISTANCE_IMPLEMENTATION(selected, dimension, chunksize,         \
                                       ImplementerClass)                       \
  do {                                                                         \
    decltype(selected) __returned_implementer;                                 \
    switch (dimension % chunksize) {                                           \
    case 0:                                                                    \
      __returned_implementer = std::make_shared<ImplementerClass>(dimension);  \
      break;                                                                   \
    case 1:                                                                    \
      __returned_implementer =                                                 \
          std::make_shared<ImplementerClass>(dimension, 1);                    \
      break;                                                                   \
    default:                                                                   \
      __returned_implementer =                                                 \
          std::make_shared<ImplementerClass>(dimension, 2);                    \
      break;                                                                   \
    }                                                                          \
    selected = __returned_implementer;                                         \
  } while (false)

} // namespace flatnav