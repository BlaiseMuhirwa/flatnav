/**
 * @file search.c
 * @brief Search and add operations for FlatNav C API
 *
 * Implements the beam search algorithm for approximate nearest neighbor
 * search, and the graph construction algorithm for adding vectors.
 */

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include "internal/structures.h"

/*============================================================================
 * Internal Index Access (defined in index.c)
 *============================================================================*/

/* Access internal index fields - these match the struct in index.c */
static inline uint32_t fn__index_M(const fn_index_t* index) {
  /* Offset after node_memory (char*) and node_memory_size (size_t) */
  const char* ptr = (const char*)index;
  ptr += sizeof(char*) + sizeof(size_t);
  return *(const uint32_t*)ptr;
}

static inline uint32_t fn__index_cur_nodes(const fn_index_t* index) {
  const char* ptr = (const char*)index;
  ptr += sizeof(char*) + sizeof(size_t) + 4 * sizeof(uint32_t);
  return *(const uint32_t*)ptr;
}

static inline fn_distance_t* fn__index_distance(const fn_index_t* index) {
  const char* ptr = (const char*)index;
  ptr += sizeof(char*) + sizeof(size_t) + 5 * sizeof(uint32_t);
  return *(fn_distance_t* const*)ptr;
}

static inline fn__visited_pool_t* fn__index_visited_pool(const fn_index_t* index) {
  const char* ptr = (const char*)index;
  /* Skip to visited_pool - this is platform dependent, so use a safer approach */
  /* For now, assume we can access it - in practice, these would be proper accessors */
  (void)ptr;
  return NULL; /* Will be passed explicitly */
}

/*============================================================================
 * Simple Random Number Generator
 *============================================================================*/

static uint32_t g_rand_state = 12345;

static uint32_t fn__rand(void) {
  g_rand_state = g_rand_state * 1103515245 + 12345;
  return (g_rand_state >> 16) & 0x7fff;
}

static uint32_t fn__rand_range(uint32_t max) {
  if (max == 0)
    return 0;
  return fn__rand() % max;
}

/*============================================================================
 * Threading Support (must match index.c)
 *============================================================================*/

#ifndef FN_NO_THREADS
#ifdef _WIN32
#include <windows.h>
typedef CRITICAL_SECTION fn__mutex_t;
#define fn__mutex_lock(m) EnterCriticalSection(m)
#define fn__mutex_unlock(m) LeaveCriticalSection(m)
#else
#include <pthread.h>
typedef pthread_mutex_t fn__mutex_t;
#define fn__mutex_lock(m) pthread_mutex_lock(m)
#define fn__mutex_unlock(m) pthread_mutex_unlock(m)
#endif
#else
typedef int fn__mutex_t;
#define fn__mutex_lock(m) ((void)(m))
#define fn__mutex_unlock(m) ((void)(m))
#endif

/*============================================================================
 * Internal Search Structure (to access index internals)
 *============================================================================*/

/* Mirror the index structure for internal access - MUST match index.c exactly! */
typedef struct fn__index_internal_s {
  char* node_memory;
  size_t node_memory_size;
  uint32_t M;
  uint32_t data_size_bytes;
  uint32_t node_size_bytes;
  uint32_t max_node_count;
  uint32_t cur_num_nodes;
  fn_distance_t* distance;
  int owns_distance;
  uint64_t distance_computations;
  uint64_t metric_hops;
  uint8_t collect_stats;
#ifndef FN_NO_THREADS
  fn__mutex_t* node_mutexes;
  fn__mutex_t global_lock;
  fn__visited_pool_t* visited_pool;
  uint32_t num_threads;
#else
  fn__visited_pool_t* visited_pool;
#endif
} fn__index_internal_t;

static inline void* fn__get_data(const fn__index_internal_t* idx, fn_node_id_t node_id) {
  return idx->node_memory + (size_t)node_id * idx->node_size_bytes;
}

static inline uint32_t* fn__get_links(const fn__index_internal_t* idx, fn_node_id_t node_id) {
  char* node_ptr = idx->node_memory + (size_t)node_id * idx->node_size_bytes;
  return (uint32_t*)(node_ptr + idx->data_size_bytes);
}

static inline fn_label_t* fn__get_label(const fn__index_internal_t* idx, fn_node_id_t node_id) {
  char* node_ptr = idx->node_memory + (size_t)node_id * idx->node_size_bytes;
  return (fn_label_t*)(node_ptr + idx->data_size_bytes + idx->M * sizeof(uint32_t));
}

/*============================================================================
 * Beam Search Implementation
 *============================================================================*/

static fn_error_t fn__beam_search(const fn__index_internal_t* idx, fn__visited_set_t* visited,
                                  fn__pqueue_t* candidates, /* Min-heap */
                                  fn__pqueue_t* results,    /* Max-heap */
                                  const void* query, fn_node_id_t entry_node, uint32_t ef, uint32_t k) {
  fn__pqueue_clear(candidates);
  fn__pqueue_clear(results);
  fn__visited_set_reset(visited);

  /* Compute distance to entry node */
  float entry_dist = fn_distance_compute(idx->distance, query, fn__get_data(idx, entry_node));

  fn__pqueue_push(candidates, entry_dist, entry_node);
  fn__pqueue_push(results, entry_dist, entry_node);
  fn__visited_set_check_and_set(visited, entry_node);

  while (!fn__pqueue_is_empty(candidates)) {
    float cand_dist;
    uint32_t cand_id;
    fn__pqueue_pop(candidates, &cand_dist, &cand_id);

    /* Check if we can stop early: if best candidate is worse than our
         * ef-th best result, we won't find anything better */
    float worst_result = fn__pqueue_top_distance(results);
    if (cand_dist > worst_result && fn__pqueue_size(results) >= ef) {
      break;
    }

    /* Explore neighbors */
    const uint32_t* links = fn__get_links(idx, cand_id);
    for (uint32_t i = 0; i < idx->M; i++) {
      uint32_t neighbor_id = links[i];
      if (neighbor_id == UINT32_MAX) {
        break; /* No more neighbors */
      }

      if (fn__visited_set_check_and_set(visited, neighbor_id)) {
        continue; /* Already visited */
      }

      float neighbor_dist = fn_distance_compute(idx->distance, query, fn__get_data(idx, neighbor_id));

      /* Standard HNSW: add to candidates and results with filtering */
      float worst_result_dist = fn__pqueue_top_distance(results);

      if (fn__pqueue_size(results) < ef || neighbor_dist < worst_result_dist) {
        fn__pqueue_push(candidates, neighbor_dist, neighbor_id);

        if (fn__pqueue_size(results) < ef) {
          fn__pqueue_push(results, neighbor_dist, neighbor_id);
        } else {
          fn__pqueue_try_push(results, neighbor_dist, neighbor_id);
        }
      }
    }
  }

  return FN_ERR_OK;
}

/*============================================================================
 * Initialize Search Entry Points
 *============================================================================*/

static fn_node_id_t fn__find_entry_node(const fn__index_internal_t* idx, const void* query,
                                        uint32_t num_initializations) {
  if (idx->cur_num_nodes == 0) {
    return 0;
  }

  if (num_initializations == 0) {
    num_initializations = 1;
  }

  /* Use strided sampling like C++ implementation */
  uint32_t step_size = idx->cur_num_nodes / num_initializations;
  if (step_size == 0)
    step_size = 1;

  fn_node_id_t best_node = 0;
  float best_dist = fn_distance_compute(idx->distance, query, fn__get_data(idx, 0));

  /* Sample entry points at regular intervals and pick the closest */
  for (fn_node_id_t node = step_size; node < idx->cur_num_nodes; node += step_size) {
    float dist = fn_distance_compute(idx->distance, query, fn__get_data(idx, node));
    if (dist < best_dist) {
      best_dist = dist;
      best_node = node;
    }
  }

  return best_node;
}

/*============================================================================
 * Public Search API
 *============================================================================*/

FN_API fn_error_t fn_index_search(const fn_index_t* index, const void* query, fn_result_t* results,
                                  const fn_search_params_t* params) {
  if (index == NULL || query == NULL || results == NULL || params == NULL) {
    return FN_ERR_NULL_PTR;
  }

  const fn__index_internal_t* idx = (const fn__index_internal_t*)index;

  if (idx->cur_num_nodes == 0) {
    /* Empty index - return empty results */
    for (uint32_t i = 0; i < params->k; i++) {
      results[i].distance = 1e30f;
      results[i].label = -1;
    }
    return FN_ERR_OK;
  }

  /* Create temporary resources */
  fn__visited_set_t* visited = NULL;
  fn__pqueue_t* candidates = NULL;
  fn__pqueue_t* result_pq = NULL;

  uint32_t ef = params->ef_search > params->k ? params->ef_search : params->k;

  visited = fn__visited_set_create(idx->cur_num_nodes);
  candidates = fn__pqueue_create(ef * 2 + 100, 1); /* Min-heap */
  result_pq = fn__pqueue_create(ef, 0);            /* Max-heap */

  if (visited == NULL || candidates == NULL || result_pq == NULL) {
    fn__visited_set_destroy(visited);
    fn__pqueue_destroy(candidates);
    fn__pqueue_destroy(result_pq);
    return FN_ERR_OUT_OF_MEMORY;
  }

  /* Find entry point */
  fn_node_id_t entry = fn__find_entry_node(idx, query, params->num_initializations);

  /* Run beam search */
  fn__beam_search(idx, visited, candidates, result_pq, query, entry, ef, params->k);

  /* Extract top-k results */
  uint32_t total_results = fn__pqueue_size(result_pq);

  /* Results are in max-heap order (worst first), so extract ALL and reverse */
  float* temp_dists = (float*)malloc(total_results * sizeof(float));
  uint32_t* temp_ids = (uint32_t*)malloc(total_results * sizeof(uint32_t));

  if (temp_dists == NULL || temp_ids == NULL) {
    free(temp_dists);
    free(temp_ids);
    fn__visited_set_destroy(visited);
    fn__pqueue_destroy(candidates);
    fn__pqueue_destroy(result_pq);
    return FN_ERR_OUT_OF_MEMORY;
  }

  /* Pop all elements (worst to best order) */
  for (uint32_t i = 0; i < total_results; i++) {
    fn__pqueue_pop(result_pq, &temp_dists[i], &temp_ids[i]);
  }

  /* Take top-k from the END (best results) and reverse into output */
  uint32_t k = params->k < total_results ? params->k : total_results;
  for (uint32_t i = 0; i < k; i++) {
    /* Best results are at the end of temp arrays after popping from max-heap */
    uint32_t src_idx = total_results - 1 - i;
    results[i].distance = temp_dists[src_idx];
    results[i].label = *fn__get_label(idx, temp_ids[src_idx]);
  }

  /* Fill remaining slots with invalid results */
  for (uint32_t i = k; i < params->k; i++) {
    results[i].distance = 1e30f;
    results[i].label = -1;
  }

  /* Cleanup */
  free(temp_dists);
  free(temp_ids);
  fn__visited_set_destroy(visited);
  fn__pqueue_destroy(candidates);
  fn__pqueue_destroy(result_pq);

  return FN_ERR_OK;
}

FN_API fn_error_t fn_index_search_with_ctx(const fn_index_t* index, fn_search_ctx_t* ctx, const void* query,
                                           fn_result_t* results, const fn_search_params_t* params) {
  /* For now, just call the regular search */
  (void)ctx;
  return fn_index_search(index, query, results, params);
}

FN_API fn_error_t fn_index_search_batch(const fn_index_t* index, const void* queries, uint32_t num_queries,
                                        fn_result_t* results, const fn_search_params_t* params) {
  if (index == NULL || queries == NULL || results == NULL || params == NULL) {
    return FN_ERR_NULL_PTR;
  }

  const fn__index_internal_t* idx = (const fn__index_internal_t*)index;
  size_t query_size = idx->distance->data_size_bytes;

  for (uint32_t q = 0; q < num_queries; q++) {
    const char* query = (const char*)queries + q * query_size;
    fn_result_t* query_results = results + q * params->k;

    fn_error_t err = fn_index_search(index, query, query_results, params);
    if (err != FN_ERR_OK) {
      return err;
    }
  }

  return FN_ERR_OK;
}

/*============================================================================
 * Neighbor Selection Heuristic
 *============================================================================*/

static void fn__select_neighbors(const fn__index_internal_t* idx,
                                 fn__pqueue_t* candidates, /* Min-heap of candidates */
                                 uint32_t* out_neighbors, uint32_t M, uint32_t* out_count) {
  *out_count = 0;

  while (!fn__pqueue_is_empty(candidates) && *out_count < M) {
    float dist;
    uint32_t node_id;
    fn__pqueue_pop(candidates, &dist, &node_id);

    /* Simple selection: just take the closest M neighbors */
    /* TODO: Implement heuristic neighbor selection */
    out_neighbors[*out_count] = node_id;
    (*out_count)++;
  }
}

/*============================================================================
 * Add Operations
 *============================================================================*/

FN_API fn_error_t fn_index_add(fn_index_t* index, const void* data, fn_label_t label,
                               uint32_t ef_construction, uint32_t num_initializations) {
  if (index == NULL || data == NULL) {
    return FN_ERR_NULL_PTR;
  }

  fn__index_internal_t* idx = (fn__index_internal_t*)index;

  /* Find entry point OUTSIDE the lock (read-only on existing nodes) */
  fn_node_id_t entry = 0;
  uint32_t cur_nodes = idx->cur_num_nodes; /* Snapshot for entry point search */
  if (cur_nodes > 0) {
    entry = fn__find_entry_node(idx, data, num_initializations);
  }

  /* Lock global mutex for node allocation (thread-safe) */
#ifndef FN_NO_THREADS
  fn__mutex_lock(&idx->global_lock);
#endif

  if (idx->cur_num_nodes >= idx->max_node_count) {
#ifndef FN_NO_THREADS
    fn__mutex_unlock(&idx->global_lock);
#endif
    return FN_ERR_INDEX_FULL;
  }

  fn_node_id_t new_node_id = idx->cur_num_nodes;

  /* Allocate and store the node data */
  void* node_data = fn__get_data(idx, new_node_id);
  if (idx->distance->ops.transform) {
    idx->distance->ops.transform(node_data, data, idx->distance->dimension, idx->distance->context);
  } else {
    memcpy(node_data, data, idx->data_size_bytes);
  }

  /* Store label */
  fn_label_t* node_label = fn__get_label(idx, new_node_id);
  *node_label = label;

  idx->cur_num_nodes++;

#ifndef FN_NO_THREADS
  fn__mutex_unlock(&idx->global_lock);
#endif

  /* If this is the first node, we're done */
  if (new_node_id == 0) {
    return FN_ERR_OK;
  }

  /* Entry point was already found above (before node allocation) */

  /* Acquire visited set from pool (avoids malloc on every add) */
  fn__visited_set_t* visited = fn__visited_pool_acquire(idx->visited_pool);

  /* Create temporary priority queues (TODO: pool these too) */
  uint32_t ef = ef_construction > idx->M ? ef_construction : idx->M;
  fn__pqueue_t* candidates = fn__pqueue_create(ef * 2 + 100, 1); /* Min-heap */
  fn__pqueue_t* result_pq = fn__pqueue_create(ef, 0);            /* Max-heap */

  if (visited == NULL || candidates == NULL || result_pq == NULL) {
    if (visited)
      fn__visited_pool_release(idx->visited_pool, visited);
    fn__pqueue_destroy(candidates);
    fn__pqueue_destroy(result_pq);
    return FN_ERR_OUT_OF_MEMORY;
  }

  /* Search for nearest neighbors */
  fn__beam_search((const fn__index_internal_t*)idx, visited, candidates, result_pq, data, entry, ef, idx->M);

  /* Extract neighbors - pop ALL from max-heap (worst first), then take best M/2
     * Note: C++ implementation uses M/2 for selection to allow back-connections
     * to fill the remaining slots */
  uint32_t selection_M = idx->M / 2;
  if (selection_M == 0)
    selection_M = 1;

  uint32_t total_results = fn__pqueue_size(result_pq);
  uint32_t* temp_ids = (uint32_t*)malloc(total_results * sizeof(uint32_t));
  if (temp_ids == NULL) {
    fn__visited_set_destroy(visited);
    fn__pqueue_destroy(candidates);
    fn__pqueue_destroy(result_pq);
    return FN_ERR_OUT_OF_MEMORY;
  }

  /* Pop all (worst to best order from max-heap) */
  for (uint32_t i = 0; i < total_results; i++) {
    float dist;
    fn__pqueue_pop(result_pq, &dist, &temp_ids[i]);
  }

  /* Take best M/2 from the END of the array (they were popped last) */
  uint32_t neighbor_count = total_results < selection_M ? total_results : selection_M;
  uint32_t* neighbors = fn__get_links(idx, new_node_id);

  for (uint32_t i = 0; i < neighbor_count; i++) {
    /* Best neighbors are at the end of temp_ids */
    neighbors[i] = temp_ids[total_results - 1 - i];
  }

  free(temp_ids);

  /* Back-connect: add new node as neighbor to selected nodes */
  for (uint32_t i = 0; i < neighbor_count; i++) {
    uint32_t neighbor_id = neighbors[i];

#ifndef FN_NO_THREADS
    /* Lock neighbor node for thread-safe back-connection */
    fn__mutex_lock(&idx->node_mutexes[neighbor_id]);
#endif

    uint32_t* neighbor_links = fn__get_links(idx, neighbor_id);

    /* Find empty slot or replace worst neighbor */
    int added = 0;
    for (uint32_t j = 0; j < idx->M; j++) {
      if (neighbor_links[j] == UINT32_MAX) {
        neighbor_links[j] = new_node_id;
        added = 1;
        break;
      }
    }

    if (!added) {
      /* All slots full - find worst neighbor and potentially replace */
      const void* neighbor_data = fn__get_data(idx, neighbor_id);
      float new_dist = fn_distance_compute(idx->distance, neighbor_data, fn__get_data(idx, new_node_id));

      float worst_dist = 0;
      uint32_t worst_idx = 0;

      for (uint32_t j = 0; j < idx->M; j++) {
        float d = fn_distance_compute(idx->distance, neighbor_data, fn__get_data(idx, neighbor_links[j]));
        if (d > worst_dist) {
          worst_dist = d;
          worst_idx = j;
        }
      }

      if (new_dist < worst_dist) {
        neighbor_links[worst_idx] = new_node_id;
      }
    }

#ifndef FN_NO_THREADS
    fn__mutex_unlock(&idx->node_mutexes[neighbor_id]);
#endif
  }

  /* Cleanup - return visited set to pool, destroy priority queues */
  fn__visited_pool_release(idx->visited_pool, visited);
  fn__pqueue_destroy(candidates);
  fn__pqueue_destroy(result_pq);

  return FN_ERR_OK;
}

FN_API fn_error_t fn_index_add_batch(fn_index_t* index, const void* data, const fn_label_t* labels,
                                     uint32_t count, uint32_t ef_construction, uint32_t num_initializations) {
  if (index == NULL || data == NULL || labels == NULL) {
    return FN_ERR_NULL_PTR;
  }

  fn__index_internal_t* idx = (fn__index_internal_t*)index;
  size_t data_size = idx->distance->data_size_bytes;

  for (uint32_t i = 0; i < count; i++) {
    const char* vec = (const char*)data + i * data_size;
    fn_error_t err = fn_index_add(index, vec, labels[i], ef_construction, num_initializations);
    if (err != FN_ERR_OK) {
      return err;
    }
  }

  return FN_ERR_OK;
}
