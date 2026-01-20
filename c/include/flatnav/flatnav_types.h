/**
 * @file flatnav_types.h
 * @brief Core types, enums, and error codes for FlatNav C API
 *
 * This header defines all fundamental types used throughout the FlatNav
 * library, including opaque pointer types, error codes, and configuration
 * structures.
 */

#ifndef FLATNAV_TYPES_H
#define FLATNAV_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Version Information
 *============================================================================*/

#define FN_VERSION_MAJOR 1
#define FN_VERSION_MINOR 0
#define FN_VERSION_PATCH 0

/*============================================================================
 * API Visibility Macros
 *============================================================================*/

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef FN_BUILD_SHARED
#define FN_API __declspec(dllexport)
#else
#define FN_API __declspec(dllimport)
#endif
#else
#if __GNUC__ >= 4
#define FN_API __attribute__((visibility("default")))
#else
#define FN_API
#endif
#endif

/* Static library builds */
#ifdef FN_BUILD_STATIC
#undef FN_API
#define FN_API
#endif

/*============================================================================
 * Opaque Pointer Types
 *============================================================================*/

/**
 * @brief Opaque index handle
 *
 * Represents a nearest neighbor search index. Created with fn_index_create()
 * and destroyed with fn_index_destroy().
 */
typedef struct fn_index_s fn_index_t;

/**
 * @brief Opaque distance function handle
 *
 * Represents a distance metric (L2, inner product, etc.) with associated
 * data type. Created with fn_distance_create_*() functions.
 */
typedef struct fn_distance_s fn_distance_t;

/**
 * @brief Thread-local search context
 *
 * Contains per-thread state for search operations (visited set, priority
 * queues). Use for concurrent search from multiple threads.
 */
typedef struct fn_search_ctx_s fn_search_ctx_t;

/*============================================================================
 * Basic Types
 *============================================================================*/

/** @brief Node identifier within the index */
typedef uint32_t fn_node_id_t;

/** @brief External label for vectors (user-provided identifier) */
#ifndef FN_LABEL_TYPE
typedef int32_t fn_label_t;
#else
typedef FN_LABEL_TYPE fn_label_t;
#endif

/*============================================================================
 * Enumerations
 *============================================================================*/

/**
 * @brief Supported vector element data types
 */
typedef enum fn_data_type_e {
  FN_DATA_FLOAT32 = 0,    /**< 32-bit floating point */
  FN_DATA_INT8 = 1,       /**< 8-bit signed integer */
  FN_DATA_UINT8 = 2,      /**< 8-bit unsigned integer */
  FN_DATA_INT16 = 3,      /**< 16-bit signed (fixed-point on DPU) */
  FN_DATA_INT32 = 4,      /**< 32-bit signed (accumulator) */
  FN_DATA_UNDEFINED = 255 /**< Undefined/invalid type */
} fn_data_type_t;

/**
 * @brief Distance metric types
 */
typedef enum fn_metric_type_e {
  FN_METRIC_L2 = 0, /**< Squared L2 (Euclidean) distance */
  FN_METRIC_IP = 1  /**< Inner product distance (1 - dot product) */
} fn_metric_type_t;

/**
 * @brief Error codes returned by FlatNav functions
 */
typedef enum fn_error_e {
  FN_ERR_OK = 0,                  /**< Success */
  FN_ERR_NULL_PTR = -1,           /**< NULL pointer argument */
  FN_ERR_INVALID_ARG = -2,        /**< Invalid argument value */
  FN_ERR_OUT_OF_MEMORY = -3,      /**< Memory allocation failed */
  FN_ERR_INDEX_FULL = -4,         /**< Index has reached max capacity */
  FN_ERR_IO_ERROR = -5,           /**< File I/O error */
  FN_ERR_INVALID_FORMAT = -6,     /**< Invalid file format */
  FN_ERR_DIMENSION_MISMATCH = -7, /**< Vector dimension mismatch */
  FN_ERR_NOT_SUPPORTED = -8,      /**< Operation not supported */
  FN_ERR_NOT_INITIALIZED = -9,    /**< Library not initialized */
  FN_ERR_INTERNAL = -99           /**< Internal error */
} fn_error_t;

/*============================================================================
 * Configuration Structures
 *============================================================================*/

/**
 * @brief Index creation configuration
 */
typedef struct fn_index_config_s {
  uint32_t dimension;          /**< Vector dimension */
  uint32_t max_node_count;     /**< Maximum number of vectors */
  uint32_t max_edges_per_node; /**< M parameter (edges per node) */
  fn_metric_type_t metric;     /**< Distance metric */
  fn_data_type_t data_type;    /**< Vector element data type */
  uint8_t collect_stats;       /**< Boolean: collect statistics */
  uint8_t _reserved[3];        /**< Padding for alignment */
} fn_index_config_t;

/**
 * @brief Search parameters
 */
typedef struct fn_search_params_s {
  uint32_t k;                   /**< Number of neighbors to return */
  uint32_t ef_search;           /**< Search beam width */
  uint32_t num_initializations; /**< Random initialization count */
} fn_search_params_t;

/**
 * @brief Single search result (distance-label pair)
 */
typedef struct fn_result_s {
  float distance;   /**< Distance to the neighbor */
  fn_label_t label; /**< Label of the neighbor */
} fn_result_t;

/*============================================================================
 * Utility Functions
 *============================================================================*/

/**
 * @brief Get human-readable error message
 * @param error Error code
 * @return Static string describing the error
 */
FN_API const char* fn_error_string(fn_error_t error);

/**
 * @brief Get size in bytes for a data type
 * @param dtype Data type
 * @return Size in bytes, or 0 if invalid
 */
FN_API size_t fn_data_type_size(fn_data_type_t dtype);

/**
 * @brief Get string name of a data type
 * @param dtype Data type
 * @return Static string name
 */
FN_API const char* fn_data_type_name(fn_data_type_t dtype);

/**
 * @brief Get string name of a metric type
 * @param metric Metric type
 * @return Static string name
 */
FN_API const char* fn_metric_type_name(fn_metric_type_t metric);

#ifdef __cplusplus
}
#endif

#endif /* FLATNAV_TYPES_H */
