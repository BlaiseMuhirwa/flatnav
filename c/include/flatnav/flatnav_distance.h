/**
 * @file flatnav_distance.h
 * @brief Distance function interface for FlatNav C API
 *
 * This header defines the distance function abstraction, replacing the C++
 * CRTP pattern with function pointers for C compatibility.
 */

#ifndef FLATNAV_DISTANCE_H
#define FLATNAV_DISTANCE_H

#include "flatnav_config.h"
#include "flatnav_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Distance Function Types
 *============================================================================*/

/**
 * @brief Distance computation function signature
 *
 * @param x First vector
 * @param y Second vector
 * @param dimension Vector dimension
 * @param context Optional context (e.g., for quantization parameters)
 * @return Distance value (lower = more similar for L2, higher for IP)
 */
typedef float (*fn_distance_func_t)(const void* x, const void* y, uint32_t dimension, void* context);

/**
 * @brief Data transformation function signature
 *
 * Used for preprocessing vectors before storage (e.g., normalization,
 * quantization).
 *
 * @param dest Destination buffer for transformed data
 * @param src Source vector data
 * @param dimension Vector dimension
 * @param context Optional context
 */
typedef void (*fn_transform_func_t)(void* dest, const void* src, uint32_t dimension, void* context);

/**
 * @brief Distance operations table (vtable-like structure)
 */
typedef struct fn_distance_ops_s {
  fn_distance_func_t compute;      /**< Symmetric distance computation */
  fn_distance_func_t compute_asym; /**< Asymmetric (query vs db) distance */
  fn_transform_func_t transform;   /**< Data transformation for storage */
  void (*destroy)(void* context);  /**< Free context resources */
} fn_distance_ops_t;

/*============================================================================
 * Distance Handle Structure (Opaque Implementation)
 *============================================================================*/

/**
 * @brief Internal structure for distance handle
 *
 * Users should treat fn_distance_t* as an opaque pointer. This definition
 * is exposed for internal use and size calculations.
 */
struct fn_distance_s {
  fn_distance_ops_t ops;    /**< Function pointer table */
  fn_metric_type_t metric;  /**< Metric type (L2, IP) */
  fn_data_type_t data_type; /**< Vector element type */
  uint32_t dimension;       /**< Vector dimension */
  uint32_t data_size_bytes; /**< Size of transformed data */
  void* context;            /**< Implementation-specific data */
};

/*============================================================================
 * Distance Creation Functions
 *============================================================================*/

/**
 * @brief Create squared L2 (Euclidean) distance function
 *
 * @param[out] out_distance Pointer to receive the distance handle
 * @param dimension Vector dimension
 * @param data_type Vector element data type
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_distance_create_l2(fn_distance_t** out_distance, uint32_t dimension,
                                        fn_data_type_t data_type);

/**
 * @brief Create inner product distance function
 *
 * Inner product distance is computed as (1 - dot_product) so that lower
 * values indicate more similar vectors.
 *
 * @param[out] out_distance Pointer to receive the distance handle
 * @param dimension Vector dimension
 * @param data_type Vector element data type
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_distance_create_ip(fn_distance_t** out_distance, uint32_t dimension,
                                        fn_data_type_t data_type);

#ifdef FN_DPU_TARGET
/**
 * @brief Create L2 distance with fixed-point arithmetic (DPU only)
 *
 * @param[out] out_distance Pointer to receive the distance handle
 * @param dimension Vector dimension
 * @param data_type Vector element data type (should be int8/uint8)
 * @param scale_bits Number of fractional bits for fixed-point
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_distance_create_l2_fixed(fn_distance_t** out_distance, uint32_t dimension,
                                              fn_data_type_t data_type, int32_t scale_bits);
#endif

/**
 * @brief Destroy a distance handle and free resources
 *
 * @param distance Distance handle to destroy (may be NULL)
 */
FN_API void fn_distance_destroy(fn_distance_t* distance);

/*============================================================================
 * Distance Computation Inline Functions
 *============================================================================*/

/**
 * @brief Compute distance between two vectors
 *
 * @param dist Distance handle
 * @param x First vector
 * @param y Second vector
 * @return Distance value
 */
FN_INLINE float fn_distance_compute(const fn_distance_t* dist, const void* x, const void* y) {
  return dist->ops.compute(x, y, dist->dimension, dist->context);
}

/**
 * @brief Compute asymmetric distance (query vs database vector)
 *
 * Used when query vectors may have different representation than
 * stored vectors (e.g., query is float, stored is quantized).
 *
 * @param dist Distance handle
 * @param query Query vector
 * @param db Database vector
 * @return Distance value
 */
FN_INLINE float fn_distance_compute_asymmetric(const fn_distance_t* dist, const void* query, const void* db) {
  if (dist->ops.compute_asym) {
    return dist->ops.compute_asym(query, db, dist->dimension, dist->context);
  }
  return dist->ops.compute(query, db, dist->dimension, dist->context);
}

/**
 * @brief Transform vector data for storage
 *
 * @param dist Distance handle
 * @param dest Destination buffer (must be at least data_size_bytes)
 * @param src Source vector
 */
FN_INLINE void fn_distance_transform(const fn_distance_t* dist, void* dest, const void* src) {
  if (dist->ops.transform) {
    dist->ops.transform(dest, src, dist->dimension, dist->context);
  }
}

/*============================================================================
 * Distance Property Accessors
 *============================================================================*/

/**
 * @brief Get vector dimension
 */
FN_INLINE uint32_t fn_distance_dimension(const fn_distance_t* dist) {
  return dist->dimension;
}

/**
 * @brief Get size of transformed data in bytes
 */
FN_INLINE uint32_t fn_distance_data_size(const fn_distance_t* dist) {
  return dist->data_size_bytes;
}

/**
 * @brief Get metric type
 */
FN_INLINE fn_metric_type_t fn_distance_metric(const fn_distance_t* dist) {
  return dist->metric;
}

/**
 * @brief Get data type
 */
FN_INLINE fn_data_type_t fn_distance_data_type(const fn_distance_t* dist) {
  return dist->data_type;
}

#ifdef __cplusplus
}
#endif

#endif /* FLATNAV_DISTANCE_H */
