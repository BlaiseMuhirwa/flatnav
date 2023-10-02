#pragma once

#include <cereal/access.hpp>
#include <cmath>
#include <flatnav/DistanceInterface.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <memory>
#include <utility>
#include <variant>

#ifdef _OPENMP
#include <omp.h>
#endif

using flatnav::DistanceInterface;
using flatnav::InnerProductDistance;
using flatnav::METRIC_TYPE;
using flatnav::SquaredL2Distance;

namespace flatnav::quantization {

/**
 * Implementation of the locally-adaptive vector quantization scheme
 * described in https://arxiv.org/pdf/2304.04759.pdf
 *
 */
class LowPrecisionQuantizer : public DistanceInterface<LowPrecisionQuantizer> {
  friend class DistanceInterface<LowPrecisionQuantizer>;

public:
  LowPrecisionQuantizer() = default;

  LowPrecisionQuantizer(uint32_t num_bits, uint32_t dim,
                        METRIC_TYPE metric_type)
      : _num_bits(num_bits), _dimension(dim), _is_trained(false),
        _metric_type(metric_type) {

    if (num_bits != 16 && num_bits != 8) {
      throw std::invalid_argument(
          std::to_string(num_bits) +
          "-bit quantization is not supported."
          "LowPrecisionQuantizer: only 8 and 16 bit quantization is "
          "supported.");
    }
    _data_size_bytes = dim * (num_bits / 8);

    if (metric_type == METRIC_TYPE::EUCLIDEAN) {
      _distance = SquaredL2Distance(dim);
    } else if (metric_type == METRIC_TYPE::INNER_PRODUCT) {
      _distance = InnerProductDistance(dim);
    }
  }

  void train(const float *vectors, uint64_t num_vectors) {
    if (_is_trained) {
      return;
    }
    _mean_vector.resize(_dimension);

#pragma omp parallel for default(none) shared(vectors, num_vectors)
    for (uint64_t vec_index = 0; vec_index < num_vectors; vec_index++) {
      for (uint32_t dim_index = 0; dim_index < _dimension; dim_index++) {
#pragma omp atomic
        _mean_vector[dim_index] += vectors[vec_index * _dimension + dim_index];
      }
    }
    for (uint32_t dim_index = 0; dim_index < _dimension; dim_index++) {
      _mean_vector[dim_index] /= num_vectors;
    }

    _is_trained = true;
  }

private:
  size_t _num_bits;
  size_t _dimension;
  size_t _data_size_bytes;

  bool _is_trained;
  METRIC_TYPE _metric_type;

  // This is a vector along the dimensions of the entire vectors.
  // mu_i is the mean of the ith dimension.
  std::vector<float> _mean_vector;

  std::variant<SquaredL2Distance, InnerProductDistance> _distance;

  friend class cereal::access;

  template <typename Archive> void serialize(Archive &ar) {
    ar(_num_bits, _dimension, _data_size_bytes, _is_trained, _metric_type,
       _mean_vector);

    if (Archive::is_loading::value) {
      if (_metric_type == METRIC_TYPE::EUCLIDEAN) {
        _distance = SquaredL2Distance(_dimension);
      } else if (_metric_type == METRIC_TYPE::INNER_PRODUCT) {
        _distance = InnerProductDistance(_dimension);
      }
    }
  }

  /**
   * @brief Computes the upper and lower limits of the quantization range.
   * for a given vector.
   *
   * @param vector data vector of size _dimension
   */
  std::pair<float, float> getMinMax(const float *vector) const {
    assert(_is_trained);

    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();

    for (uint32_t dim_index = 0; dim_index < _dimension; dim_index++) {
      auto diff = vector[dim_index] - _mean_vector[dim_index];
      if (diff < min) {
        min = diff;
      }
      if (diff > max) {
        max = diff;
      }
    }
    return {min, max};
  }

  /**
   * @brief Quantizes a given vector following the locally-adaptive quantization
   * scheme. quantized_vector must be pre-allocated.
   *
   * @param vector
   * @param quantized_vector
   */
  template <typename T>
  void quantizeVector(const float *vector, T *quantized_vector) const {
    assert(_is_trained);
    assert(sizeof(T) == _num_bits / 8);

    auto [min, max] = getMinMax(vector);

    float delta = (max - min) / ((1 << _num_bits) - 1);

    for (uint32_t dim_index = 0; dim_index < _dimension; dim_index++) {
      auto first_term = (vector[dim_index] - min) / delta;
      first_term += 0.5;
      quantized_vector[dim_index] =
          static_cast<T>(delta * std::floor(first_term) + min);
    }
  }

  template <typename data_t>
  float distanceImpl(const void *x, const void *y,
                     bool asymmetric = false) const {
    bool metric_euclidean = _metric_type == METRIC_TYPE::EUCLIDEAN;

    if (_num_bits == 16) {
      void *x_ptr = (void *)x;

      if (asymmetric) {
        std::unique_ptr<int16_t[]> quantized_x =
            std::make_unique<int16_t[]>(_dimension);
        quantizeVector<int16_t>((float *)x, quantized_x.get());
        x_ptr = (void *)quantized_x.get();
      }
      float distance =
          metric_euclidean
              ? std::get<0>(_distance).distanceImpl<int16_t>(x_ptr, y)
              : std::get<1>(_distance).distanceImpl<int16_t>(x_ptr, y);
      return distance;

    } else if (_num_bits == 8) {
      void *x_ptr = (void *)x;
      if (asymmetric) {
        std::unique_ptr<int8_t[]> quantized_x =
            std::make_unique<int8_t[]>(_dimension);
        quantizeVector<int8_t>((float *)x, quantized_x.get());
        x_ptr = (void *)quantized_x.get();
      }

      float distance =
          metric_euclidean
              ? std::get<0>(_distance).distanceImpl<int8_t>(x_ptr, y)
              : std::get<1>(_distance).distanceImpl<int8_t>(x_ptr, y);
      return distance;
    }

    throw std::runtime_error("LowPrecisionQuantizer: unsupported bit width");
  }

  inline size_t getDimension() const { return _dimension; }

  size_t dataSizeImpl() { return _data_size_bytes; }

  void transformDataImpl(void *destination, const void *src) {

    if (_num_bits == 16) {
      int16_t *quantized_vector = new int16_t[_dimension]();
      quantizeVector<int16_t>((float *)src, quantized_vector);
      std::memcpy(destination, quantized_vector, _data_size_bytes);

      delete[] quantized_vector;
    } else if (_num_bits == 8) {
      int8_t *quantized_vector = new int8_t[_dimension]();
      quantizeVector<int8_t>((float *)src, quantized_vector);
      std::memcpy(destination, quantized_vector, _data_size_bytes);

      delete[] quantized_vector;
    }
  }

  void printParamsImpl() {
    std::cout << "\nLow-Precision Quantization Parameters" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    std::cout << "Dimension: " << _dimension << std::endl;
    std::cout << "Number of bits: " << _num_bits << std::endl;
    std::cout << "Data size: " << _data_size_bytes << std::endl;
    std::cout << "Is trained: " << _is_trained << std::endl;
  }
};

} // namespace flatnav::quantization
