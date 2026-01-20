/**
 * @file index.c
 * @brief Core index operations for FlatNav C API
 */

#include <stdlib.h>
#include <string.h>
#include "internal/structures.h"

/*============================================================================
 * Threading Support
 *============================================================================*/

#ifndef FN_NO_THREADS
#ifdef _WIN32
#include <windows.h>
typedef CRITICAL_SECTION fn__mutex_t;
#define fn__mutex_init(m) InitializeCriticalSection(m)
#define fn__mutex_destroy(m) DeleteCriticalSection(m)
#define fn__mutex_lock(m) EnterCriticalSection(m)
#define fn__mutex_unlock(m) LeaveCriticalSection(m)
#else
#include <pthread.h>
typedef pthread_mutex_t fn__mutex_t;
#define fn__mutex_init(m) pthread_mutex_init(m, NULL)
#define fn__mutex_destroy(m) pthread_mutex_destroy(m)
#define fn__mutex_lock(m) pthread_mutex_lock(m)
#define fn__mutex_unlock(m) pthread_mutex_unlock(m)
#endif
#else
typedef int fn__mutex_t;
#define fn__mutex_init(m) ((void)(m))
#define fn__mutex_destroy(m) ((void)(m))
#define fn__mutex_lock(m) ((void)(m))
#define fn__mutex_unlock(m) ((void)(m))
#endif

/*============================================================================
 * Index Structure Definition
 *============================================================================*/

struct fn_index_s {
  /* Memory block for all nodes */
  char* node_memory;
  size_t node_memory_size;

  /* Index parameters */
  uint32_t M;               /* Max edges per node */
  uint32_t data_size_bytes; /* Size of vector data */
  uint32_t node_size_bytes; /* Total node size */
  uint32_t max_node_count;
  uint32_t cur_num_nodes;

  /* Distance function */
  fn_distance_t* distance;
  int owns_distance; /* Whether we should free distance */

  /* Statistics */
  uint64_t distance_computations;
  uint64_t metric_hops;
  uint8_t collect_stats;

  /* Thread safety */
#ifndef FN_NO_THREADS
  fn__mutex_t* node_mutexes;
  fn__mutex_t global_lock;
  fn__visited_pool_t* visited_pool;
  uint32_t num_threads;
#else
  fn__visited_pool_t* visited_pool;
#endif
};

/*============================================================================
 * Search Context Structure
 *============================================================================*/

struct fn_search_ctx_s {
  fn__visited_set_t* visited;
  fn__pqueue_t* candidates; /* Min-heap for candidates */
  fn__pqueue_t* results;    /* Max-heap for results */
  uint32_t max_k;           /* Maximum k this context supports */
  uint32_t max_ef;          /* Maximum ef this context supports */
};

/*============================================================================
 * Node Memory Layout Accessors
 *============================================================================*/

/*
 * Node layout:
 * [vector data: data_size_bytes]
 * [neighbor links: M * sizeof(uint32_t)]
 * [label: sizeof(fn_label_t)]
 */

static inline void* fn__get_node_data(const fn_index_t* index, fn_node_id_t node_id) {
  return index->node_memory + (size_t)node_id * index->node_size_bytes;
}

static inline uint32_t* fn__get_node_links(const fn_index_t* index, fn_node_id_t node_id) {
  char* node_ptr = index->node_memory + (size_t)node_id * index->node_size_bytes;
  return (uint32_t*)(node_ptr + index->data_size_bytes);
}

static inline fn_label_t* fn__get_node_label(const fn_index_t* index, fn_node_id_t node_id) {
  char* node_ptr = index->node_memory + (size_t)node_id * index->node_size_bytes;
  return (fn_label_t*)(node_ptr + index->data_size_bytes + index->M * sizeof(uint32_t));
}

/*============================================================================
 * Index Creation
 *============================================================================*/

FN_API fn_error_t fn_index_create(fn_index_t** out_index, const fn_index_config_t* config) {
  if (out_index == NULL || config == NULL) {
    return FN_ERR_NULL_PTR;
  }

  if (config->dimension == 0 || config->max_node_count == 0) {
    return FN_ERR_INVALID_ARG;
  }

  /* Create distance function */
  fn_distance_t* distance = NULL;
  fn_error_t err;

  if (config->metric == FN_METRIC_L2) {
    err = fn_distance_create_l2(&distance, config->dimension, config->data_type);
  } else if (config->metric == FN_METRIC_IP) {
    err = fn_distance_create_ip(&distance, config->dimension, config->data_type);
  } else {
    return FN_ERR_INVALID_ARG;
  }

  if (err != FN_ERR_OK) {
    return err;
  }

  err =
      fn_index_create_with_distance(out_index, distance, config->max_node_count, config->max_edges_per_node);
  if (err != FN_ERR_OK) {
    fn_distance_destroy(distance);
    return err;
  }

  (*out_index)->collect_stats = config->collect_stats;
  (*out_index)->owns_distance = 1;

  return FN_ERR_OK;
}

FN_API fn_error_t fn_index_create_with_distance(fn_index_t** out_index, fn_distance_t* distance,
                                                uint32_t max_node_count, uint32_t max_edges_per_node) {
  if (out_index == NULL || distance == NULL) {
    return FN_ERR_NULL_PTR;
  }

  if (max_node_count == 0 || max_edges_per_node == 0) {
    return FN_ERR_INVALID_ARG;
  }

  fn_index_t* index = (fn_index_t*)calloc(1, sizeof(fn_index_t));
  if (index == NULL) {
    return FN_ERR_OUT_OF_MEMORY;
  }

  index->distance = distance;
  index->owns_distance = 0; /* Caller passed it in, doesn't own by default */
  index->M = max_edges_per_node;
  index->data_size_bytes = distance->data_size_bytes;
  index->max_node_count = max_node_count;
  index->cur_num_nodes = 0;

  /* Calculate node size */
  index->node_size_bytes = index->data_size_bytes + index->M * sizeof(uint32_t) + sizeof(fn_label_t);

  /* Allocate node memory */
  index->node_memory_size = (size_t)index->node_size_bytes * max_node_count;
  index->node_memory = (char*)calloc(1, index->node_memory_size);
  if (index->node_memory == NULL) {
    free(index);
    return FN_ERR_OUT_OF_MEMORY;
  }

  /* Initialize neighbor links to UINT32_MAX (no neighbor) */
  for (uint32_t i = 0; i < max_node_count; i++) {
    uint32_t* links = fn__get_node_links(index, i);
    for (uint32_t j = 0; j < max_edges_per_node; j++) {
      links[j] = UINT32_MAX;
    }
  }

#ifndef FN_NO_THREADS
  /* Allocate per-node mutexes */
  index->node_mutexes = (fn__mutex_t*)calloc(max_node_count, sizeof(fn__mutex_t));
  if (index->node_mutexes == NULL) {
    free(index->node_memory);
    free(index);
    return FN_ERR_OUT_OF_MEMORY;
  }

  for (uint32_t i = 0; i < max_node_count; i++) {
    fn__mutex_init(&index->node_mutexes[i]);
  }

  fn__mutex_init(&index->global_lock);
  index->num_threads = 1;
#endif

  /* Create visited set pool */
  uint32_t pool_size = 8; /* Default pool size */
  index->visited_pool = fn__visited_pool_create(max_node_count, pool_size);
  if (index->visited_pool == NULL) {
#ifndef FN_NO_THREADS
    for (uint32_t i = 0; i < max_node_count; i++) {
      fn__mutex_destroy(&index->node_mutexes[i]);
    }
    fn__mutex_destroy(&index->global_lock);
    free(index->node_mutexes);
#endif
    free(index->node_memory);
    free(index);
    return FN_ERR_OUT_OF_MEMORY;
  }

  *out_index = index;
  return FN_ERR_OK;
}

FN_API void fn_index_destroy(fn_index_t* index) {
  if (index == NULL) {
    return;
  }

#ifndef FN_NO_THREADS
  for (uint32_t i = 0; i < index->max_node_count; i++) {
    fn__mutex_destroy(&index->node_mutexes[i]);
  }
  fn__mutex_destroy(&index->global_lock);
  free(index->node_mutexes);
#endif

  fn__visited_pool_destroy(index->visited_pool);

  if (index->owns_distance && index->distance) {
    fn_distance_destroy(index->distance);
  }

  free(index->node_memory);
  free(index);
}

/*============================================================================
 * Index Properties
 *============================================================================*/

FN_API uint32_t fn_index_get_dimension(const fn_index_t* index) {
  return index ? index->distance->dimension : 0;
}

FN_API uint32_t fn_index_get_num_nodes(const fn_index_t* index) {
  return index ? index->cur_num_nodes : 0;
}

FN_API uint32_t fn_index_get_max_nodes(const fn_index_t* index) {
  return index ? index->max_node_count : 0;
}

FN_API fn_metric_type_t fn_index_get_metric(const fn_index_t* index) {
  return index ? index->distance->metric : FN_METRIC_L2;
}

FN_API fn_data_type_t fn_index_get_data_type(const fn_index_t* index) {
  return index ? index->distance->data_type : FN_DATA_UNDEFINED;
}

FN_API uint32_t fn_index_get_max_edges(const fn_index_t* index) {
  return index ? index->M : 0;
}

FN_API const fn_distance_t* fn_index_get_distance(const fn_index_t* index) {
  return index ? index->distance : NULL;
}

#ifndef FN_NO_THREADS
FN_API fn_error_t fn_index_set_num_threads(fn_index_t* index, uint32_t num_threads) {
  if (index == NULL) {
    return FN_ERR_NULL_PTR;
  }
  index->num_threads = num_threads > 0 ? num_threads : 1;
  return FN_ERR_OK;
}

FN_API uint32_t fn_index_get_num_threads(const fn_index_t* index) {
  return index ? index->num_threads : 1;
}
#endif

/*============================================================================
 * Statistics
 *============================================================================*/

FN_API uint64_t fn_index_get_distance_computations(const fn_index_t* index) {
  return index ? index->distance_computations : 0;
}

FN_API uint64_t fn_index_get_metric_hops(const fn_index_t* index) {
  return index ? index->metric_hops : 0;
}

FN_API void fn_index_reset_stats(fn_index_t* index) {
  if (index) {
    index->distance_computations = 0;
    index->metric_hops = 0;
  }
}

/*============================================================================
 * Search Context
 *============================================================================*/

FN_API fn_error_t fn_search_ctx_create(fn_search_ctx_t** out_ctx, const fn_index_t* index) {
  if (out_ctx == NULL || index == NULL) {
    return FN_ERR_NULL_PTR;
  }

  fn_search_ctx_t* ctx = (fn_search_ctx_t*)calloc(1, sizeof(fn_search_ctx_t));
  if (ctx == NULL) {
    return FN_ERR_OUT_OF_MEMORY;
  }

  /* Default sizes - can be expanded if needed */
  ctx->max_k = 100;
  ctx->max_ef = 500;

  ctx->candidates = fn__pqueue_create(ctx->max_ef, 1); /* Min-heap */
  ctx->results = fn__pqueue_create(ctx->max_ef, 0);    /* Max-heap */

  if (ctx->candidates == NULL || ctx->results == NULL) {
    fn__pqueue_destroy(ctx->candidates);
    fn__pqueue_destroy(ctx->results);
    free(ctx);
    return FN_ERR_OUT_OF_MEMORY;
  }

  ctx->visited = NULL; /* Will be acquired from pool during search */

  *out_ctx = ctx;
  return FN_ERR_OK;
}

FN_API void fn_search_ctx_destroy(fn_search_ctx_t* ctx) {
  if (ctx == NULL) {
    return;
  }

  fn__pqueue_destroy(ctx->candidates);
  fn__pqueue_destroy(ctx->results);
  /* Note: visited is returned to pool, not destroyed here */
  free(ctx);
}

/*============================================================================
 * Node Allocation
 *============================================================================*/

static void fn__allocate_node(fn_index_t* index, const void* data, fn_label_t label, fn_node_id_t node_id) {
  /* Transform and store vector data */
  void* node_data = fn__get_node_data(index, node_id);
  if (index->distance->ops.transform) {
    index->distance->ops.transform(node_data, data, index->distance->dimension, index->distance->context);
  } else {
    memcpy(node_data, data, index->data_size_bytes);
  }

  /* Store label */
  fn_label_t* node_label = fn__get_node_label(index, node_id);
  *node_label = label;

  /* Links are already initialized to UINT32_MAX */
}

/*============================================================================
 * Graph Reordering (Stubs)
 *============================================================================*/

FN_API fn_error_t fn_index_reorder_gorder(fn_index_t* index, int window_size) {
  (void)index;
  (void)window_size;
  /* TODO: Implement GOrder reordering */
  return FN_ERR_NOT_SUPPORTED;
}

FN_API fn_error_t fn_index_reorder_rcm(fn_index_t* index) {
  (void)index;
  /* TODO: Implement RCM reordering */
  return FN_ERR_NOT_SUPPORTED;
}
