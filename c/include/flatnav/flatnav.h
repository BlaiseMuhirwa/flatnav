/**
 * @file flatnav.h
 * @brief Main header for FlatNav C API
 *
 * FlatNav is a high-performance nearest neighbor search library.
 * This C API provides a pure C interface suitable for use in
 * environments without C++ runtime, including UPMEM DPUs.
 *
 * Basic usage:
 * @code
 * #include <flatnav/flatnav.h>
 *
 * // Initialize library
 * fn_init();
 *
 * // Create index configuration
 * fn_index_config_t config = {
 *     .dimension = 128,
 *     .max_node_count = 10000,
 *     .max_edges_per_node = 32,
 *     .metric = FN_METRIC_L2,
 *     .data_type = FN_DATA_FLOAT32,
 *     .collect_stats = 0
 * };
 *
 * // Create index
 * fn_index_t* index = NULL;
 * fn_error_t err = fn_index_create(&index, &config);
 * if (err != FN_ERR_OK) {
 *     fprintf(stderr, "Error: %s\n", fn_error_string(err));
 *     return 1;
 * }
 *
 * // Add vectors
 * for (int i = 0; i < num_vectors; i++) {
 *     fn_index_add(index, vectors[i], i, 100, 100);
 * }
 *
 * // Search
 * fn_search_params_t params = { .k = 10, .ef_search = 100, .num_initializations = 100 };
 * fn_result_t results[10];
 * fn_index_search(index, query, results, &params);
 *
 * // Cleanup
 * fn_index_destroy(index);
 * fn_cleanup();
 * @endcode
 */

#ifndef FLATNAV_H
#define FLATNAV_H

#include "flatnav_config.h"
#include "flatnav_distance.h"
#include "flatnav_index.h"
#include "flatnav_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Library Initialization
 *============================================================================*/

/**
 * @brief Initialize the FlatNav library
 *
 * Must be called before any other FlatNav functions. Safe to call
 * multiple times; subsequent calls are no-ops.
 */
FN_API void fn_init(void);

/**
 * @brief Cleanup FlatNav library resources
 *
 * Should be called when done using FlatNav. After cleanup, fn_init()
 * must be called again before using other functions.
 */
FN_API void fn_cleanup(void);

/**
 * @brief Check if library is initialized
 *
 * @return 1 if initialized, 0 otherwise
 */
FN_API int fn_is_initialized(void);

/*============================================================================
 * Version Information
 *============================================================================*/

/**
 * @brief Get version string
 *
 * @return Version string in format "major.minor.patch"
 */
FN_API const char* fn_version_string(void);

/**
 * @brief Get major version number
 */
FN_API int fn_version_major(void);

/**
 * @brief Get minor version number
 */
FN_API int fn_version_minor(void);

/**
 * @brief Get patch version number
 */
FN_API int fn_version_patch(void);

/*============================================================================
 * Build Information
 *============================================================================*/

/**
 * @brief Get build configuration string
 *
 * Returns a string describing the build configuration, including
 * enabled SIMD features.
 *
 * @return Static string describing build configuration
 */
FN_API const char* fn_build_info(void);

/**
 * @brief Check if built for DPU target
 *
 * @return 1 if DPU build, 0 otherwise
 */
FN_API int fn_is_dpu_build(void);

/**
 * @brief Check if SIMD is enabled
 *
 * @return 1 if any SIMD support is compiled in, 0 otherwise
 */
FN_API int fn_has_simd(void);

#ifdef __cplusplus
}
#endif

#endif /* FLATNAV_H */
