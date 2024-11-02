#pragma once

#include <cstdint>

namespace flatnav::util {

/**
 * @brief Enum class for data types
 * We currently support indexes of type float32, uint8 and int8.
 */
enum class DataType {
  uint8,    /** Unsigned 8-bit integer */
  uint16,   /** Unsigned 16-bit integer */
  uint32,   /** Unsigned 32-bit integer */
  uint64,   /** Unsigned 64-bit integer */
  int8,     /** Signed 8-bit integer */
  int16,    /** Signed 16-bit integer */
  int32,    /** Signed 32-bit integer */
  int64,    /** Signed 64-bit integer */
  float16,  /** 16-bit floating-point number */
  float32,  /** 32-bit floating-point number */
  float64,  /** 64-bit floating-point number */
  undefined /** Undefined data type */
};

/**
 * @brief Get a string representation of the data type
 */
inline constexpr const char* name(DataType data_type) {
  switch (data_type) {
    case DataType::uint8:
      return "uint8";
    case DataType::uint16:
      return "uint16";
    case DataType::uint32:
      return "uint32";
    case DataType::uint64:
      return "uint64";
    case DataType::int8:
      return "int8";
    case DataType::int16:
      return "int16";
    case DataType::int32:
      return "int32";
    case DataType::int64:
      return "int64";
    case DataType::float16:
      return "float16";
    case DataType::float32:
      return "float32";
    case DataType::float64:
      return "float64";
    default:
      return "undefined";
  }
}

/**
 * @brief Get the data type from a string representation
 */
inline constexpr DataType type(const std::string_view& data_type) {
  if (data_type == "uint8") {
    return DataType::uint8;
  } else if (data_type == "uint16") {
    return DataType::uint16;
  } else if (data_type == "uint32") {
    return DataType::uint32;
  } else if (data_type == "uint64") {
    return DataType::uint64;
  } else if (data_type == "int8") {
    return DataType::int8;
  } else if (data_type == "int16") {
    return DataType::int16;
  } else if (data_type == "int32") {
    return DataType::int32;
  } else if (data_type == "int64") {
    return DataType::int64;
  } else if (data_type == "float16") {
    return DataType::float16;
  } else if (data_type == "float32") {
    return DataType::float32;
  } else if (data_type == "float64") {
    return DataType::float64;
  } else {
    return DataType::undefined;
  }
}

/**
 * @brief Get the size of the data type in bytes
 */
inline constexpr size_t size(DataType data_type) {
  switch (data_type) {
    case DataType::uint8:
      return sizeof(uint8_t);
    case DataType::uint16:
      return sizeof(uint16_t);
    case DataType::uint32:
      return sizeof(uint32_t);
    case DataType::uint64:
      return sizeof(uint64_t);
    case DataType::int8:
      return sizeof(int8_t);
    case DataType::int16:
      return sizeof(int16_t);
    case DataType::int32:
      return sizeof(int32_t);
    case DataType::int64:
      return sizeof(int64_t);
    case DataType::float16:
      return sizeof(float) / 2;
    case DataType::float32:
      return sizeof(float);
    case DataType::float64:
      return sizeof(double);
    default:
      return 0;
  }
}

// Some nice template metaprogramming (TMP) to allow us to get compile-time
// distance dispatching.
template <DataType data_type>
struct type_for_data_type;

template <>
struct type_for_data_type<DataType::float32> {
  using type = float;
};
template <>
struct type_for_data_type<DataType::int8> {
  using type = int8_t;
};
template <>
struct type_for_data_type<DataType::uint8> {
  using type = uint8_t;
};

/**
 * @brief Template metaprogramming to allow compile-time distance dispatching
 * for each data type
 * This is useful for iterating over each data type in a compile-time loop.
 * One place where this is used is in python bindings to generate the Index
 * class for each one of the supported data types. Here is a simple example of
 * how to use this:
 * @code
 * struct Callable {
 *   template <DataType data_type> void operator()() {
 *     std::cout << "Data type: " << name(data_type) << std::endl;
 *   }
 * };
 * for_each_data_type<Callable>::apply(Callable());
 * // If you have multiple data types, you can pass them as template arguments
 * like this: for_each_data_type<Callable, DataType::uint8,
 * DataType::float32>::apply(Callable());
 * @endcode
 * @tparam F A callable object
 * @tparam data_types The data types to iterate over
 */
template <typename F, DataType... data_types>
struct for_each_data_type;

/**
 * @brief Template specialization for for_each_data_type when there are data
 * types to iterate over
 * @tparam F A callable object
 * @tparam data_type The current data type
 * @tparam rest The remaining data types
 */
template <typename F, DataType data_type, DataType... rest>
struct for_each_data_type<F, data_type, rest...> {
  static void apply(F&& f) {
    f.template operator()<data_type>();
    for_each_data_type<F, rest...>::apply(std::forward<F>(f));
  }
};

/**
 * @brief Template specialization for for_each_data_type when there are no data
 * types to iterate over
 * @tparam F A callable object
 */
template <typename F>
struct for_each_data_type<F> {
  static void apply(F&&) {}
};

}  // namespace flatnav::util