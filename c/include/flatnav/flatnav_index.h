/**
 * @file flatnav_index.h
 * @brief Index operations for FlatNav C API
 *
 * This header defines the main index structure and operations for
 * nearest neighbor search.
 */

#ifndef FLATNAV_INDEX_H
#define FLATNAV_INDEX_H

#include "flatnav_distance.h"
#include "flatnav_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Index Lifecycle
 *============================================================================*/

/**
 * @brief Create a new index
 *
 * @param[out] out_index Pointer to receive the index handle
 * @param config Index configuration
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_index_create(fn_index_t** out_index, const fn_index_config_t* config);

/**
 * @brief Create index with custom distance function
 *
 * Takes ownership of the distance handle.
 *
 * @param[out] out_index Pointer to receive the index handle
 * @param distance Distance handle (ownership transferred)
 * @param max_node_count Maximum number of vectors
 * @param max_edges_per_node M parameter
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_index_create_with_distance(fn_index_t** out_index, fn_distance_t* distance,
                                                uint32_t max_node_count, uint32_t max_edges_per_node);

/**
 * @brief Destroy an index and free all resources
 *
 * @param index Index handle to destroy (may be NULL)
 */
FN_API void fn_index_destroy(fn_index_t* index);

/*============================================================================
 * Index Operations
 *============================================================================*/

/**
 * @brief Add a single vector to the index
 *
 * @param index Index handle
 * @param data Vector data
 * @param label External label for the vector
 * @param ef_construction Construction beam width
 * @param num_initializations Number of random starting points
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_index_add(fn_index_t* index, const void* data, fn_label_t label,
                               uint32_t ef_construction, uint32_t num_initializations);

/**
 * @brief Add multiple vectors to the index
 *
 * Vectors are added in parallel when threading is available.
 *
 * @param index Index handle
 * @param data Contiguous array of vectors
 * @param labels Array of labels
 * @param count Number of vectors to add
 * @param ef_construction Construction beam width
 * @param num_initializations Number of random starting points
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_index_add_batch(fn_index_t* index, const void* data, const fn_label_t* labels,
                                     uint32_t count, uint32_t ef_construction, uint32_t num_initializations);

/**
 * @brief Search for k nearest neighbors
 *
 * @param index Index handle
 * @param query Query vector
 * @param[out] results Array to receive results (must have space for k results)
 * @param params Search parameters
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_index_search(const fn_index_t* index, const void* query, fn_result_t* results,
                                  const fn_search_params_t* params);

/**
 * @brief Search for k nearest neighbors for multiple queries
 *
 * @param index Index handle
 * @param queries Contiguous array of query vectors
 * @param num_queries Number of queries
 * @param[out] results Array to receive results (must have space for num_queries * k results)
 * @param params Search parameters
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_index_search_batch(const fn_index_t* index, const void* queries, uint32_t num_queries,
                                        fn_result_t* results, const fn_search_params_t* params);

/*============================================================================
 * Thread-Safe Search Context
 *============================================================================*/

/**
 * @brief Create a search context for thread-safe operations
 *
 * Each thread should have its own search context when performing
 * concurrent searches.
 *
 * @param[out] out_ctx Pointer to receive the context handle
 * @param index Index handle
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_search_ctx_create(fn_search_ctx_t** out_ctx, const fn_index_t* index);

/**
 * @brief Destroy a search context
 *
 * @param ctx Context handle to destroy (may be NULL)
 */
FN_API void fn_search_ctx_destroy(fn_search_ctx_t* ctx);

/**
 * @brief Search using a specific thread context
 *
 * @param index Index handle
 * @param ctx Search context
 * @param query Query vector
 * @param[out] results Array to receive results
 * @param params Search parameters
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_index_search_with_ctx(const fn_index_t* index, fn_search_ctx_t* ctx, const void* query,
                                           fn_result_t* results, const fn_search_params_t* params);

/*============================================================================
 * Index Properties
 *============================================================================*/

/**
 * @brief Get vector dimension
 */
FN_API uint32_t fn_index_get_dimension(const fn_index_t* index);

/**
 * @brief Get current number of vectors in the index
 */
FN_API uint32_t fn_index_get_num_nodes(const fn_index_t* index);

/**
 * @brief Get maximum capacity of the index
 */
FN_API uint32_t fn_index_get_max_nodes(const fn_index_t* index);

/**
 * @brief Get distance metric type
 */
FN_API fn_metric_type_t fn_index_get_metric(const fn_index_t* index);

/**
 * @brief Get vector data type
 */
FN_API fn_data_type_t fn_index_get_data_type(const fn_index_t* index);

/**
 * @brief Get maximum edges per node (M parameter)
 */
FN_API uint32_t fn_index_get_max_edges(const fn_index_t* index);

/**
 * @brief Get the distance handle used by this index
 */
FN_API const fn_distance_t* fn_index_get_distance(const fn_index_t* index);

/*============================================================================
 * Threading Configuration
 *============================================================================*/

#ifndef FN_NO_THREADS

/**
 * @brief Set number of threads for parallel operations
 *
 * @param index Index handle
 * @param num_threads Number of threads (0 = auto-detect)
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_index_set_num_threads(fn_index_t* index, uint32_t num_threads);

/**
 * @brief Get current number of threads
 */
FN_API uint32_t fn_index_get_num_threads(const fn_index_t* index);

#endif /* !FN_NO_THREADS */

/*============================================================================
 * Graph Reordering
 *============================================================================*/

/**
 * @brief Reorder graph using GOrder algorithm for cache locality
 *
 * @param index Index handle
 * @param window_size Window size parameter for GOrder
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_index_reorder_gorder(fn_index_t* index, int window_size);

/**
 * @brief Reorder graph using Reverse Cuthill-McKee algorithm
 *
 * @param index Index handle
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_index_reorder_rcm(fn_index_t* index);

/*============================================================================
 * Statistics
 *============================================================================*/

/**
 * @brief Get total number of distance computations performed
 */
FN_API uint64_t fn_index_get_distance_computations(const fn_index_t* index);

/**
 * @brief Get total number of graph hops during search
 */
FN_API uint64_t fn_index_get_metric_hops(const fn_index_t* index);

/**
 * @brief Reset all statistics counters
 */
FN_API void fn_index_reset_stats(fn_index_t* index);

/*============================================================================
 * Serialization
 *============================================================================*/

/**
 * @brief Save index to file
 *
 * @param index Index handle
 * @param filename Path to output file
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_index_save(const fn_index_t* index, const char* filename);

/**
 * @brief Load index from file
 *
 * @param[out] out_index Pointer to receive the index handle
 * @param filename Path to input file
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_index_load(fn_index_t** out_index, const char* filename);

/**
 * @brief Get required buffer size for serialization
 *
 * @param index Index handle
 * @return Required buffer size in bytes
 */
FN_API size_t fn_index_serialized_size(const fn_index_t* index);

/**
 * @brief Serialize index to memory buffer
 *
 * @param index Index handle
 * @param buffer Output buffer
 * @param[in,out] buffer_size Buffer size (in: available, out: used)
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_index_to_buffer(const fn_index_t* index, void* buffer, size_t* buffer_size);

/**
 * @brief Deserialize index from memory buffer
 *
 * @param[out] out_index Pointer to receive the index handle
 * @param buffer Input buffer
 * @param buffer_size Buffer size
 * @return FN_ERR_OK on success, error code otherwise
 */
FN_API fn_error_t fn_index_from_buffer(fn_index_t** out_index, const void* buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif /* FLATNAV_INDEX_H */
