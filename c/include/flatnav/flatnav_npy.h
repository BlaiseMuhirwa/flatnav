/**
 * @file flatnav_npy.h
 * @brief Simple NPY file reader for benchmarking
 */

#ifndef FLATNAV_NPY_H
#define FLATNAV_NPY_H

#include <stdint.h>
#include "flatnav_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque NPY array handle
 */
typedef struct fn_npy_array_s fn_npy_array_t;

/**
 * @brief Load an NPY file
 *
 * @param filename Path to the NPY file
 * @return Array handle or NULL on error
 */
FN_API fn_npy_array_t* fn_npy_load(const char* filename);

/**
 * @brief Free an NPY array
 *
 * @param arr Array to free
 */
FN_API void fn_npy_free(fn_npy_array_t* arr);

/**
 * @brief Get float32 data pointer
 *
 * @param arr Array handle
 * @return Pointer to float data, or NULL if array is not float
 */
FN_API float* fn_npy_data_float(fn_npy_array_t* arr);

/**
 * @brief Get int32 data pointer
 *
 * @param arr Array handle
 * @return Pointer to int data, or NULL if array is not int
 */
FN_API int32_t* fn_npy_data_int(fn_npy_array_t* arr);

/**
 * @brief Get number of rows (first dimension)
 */
FN_API uint32_t fn_npy_rows(fn_npy_array_t* arr);

/**
 * @brief Get number of columns (second dimension)
 */
FN_API uint32_t fn_npy_cols(fn_npy_array_t* arr);

#ifdef __cplusplus
}
#endif

#endif /* FLATNAV_NPY_H */
