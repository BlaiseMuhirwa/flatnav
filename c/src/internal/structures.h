/**
 * @file internal/structures.h
 * @brief Internal structure definitions shared between source files
 */

#ifndef FLATNAV_INTERNAL_STRUCTURES_H
#define FLATNAV_INTERNAL_STRUCTURES_H

#include <flatnav/flatnav.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Visited Set Structure
 *============================================================================*/

typedef struct fn__visited_set_s {
  uint8_t* markers;       /**< Marker array, one byte per node */
  uint32_t capacity;      /**< Number of nodes */
  uint8_t current_marker; /**< Current marker value */
} fn__visited_set_t;

/*============================================================================
 * Visited Set Functions
 *============================================================================*/

fn__visited_set_t* fn__visited_set_create(uint32_t capacity);
void fn__visited_set_destroy(fn__visited_set_t* vs);
void fn__visited_set_reset(fn__visited_set_t* vs);
int fn__visited_set_check_and_set(fn__visited_set_t* vs, uint32_t node_id);
int fn__visited_set_is_visited(const fn__visited_set_t* vs, uint32_t node_id);

/*============================================================================
 * Visited Set Pool
 *============================================================================*/

typedef struct fn__visited_pool_s fn__visited_pool_t;

fn__visited_pool_t* fn__visited_pool_create(uint32_t node_capacity, uint32_t pool_size);
void fn__visited_pool_destroy(fn__visited_pool_t* pool);
fn__visited_set_t* fn__visited_pool_acquire(fn__visited_pool_t* pool);
void fn__visited_pool_release(fn__visited_pool_t* pool, fn__visited_set_t* vs);

/*============================================================================
 * Priority Queue Element
 *============================================================================*/

typedef struct fn__pq_elem_s {
  float distance;
  uint32_t node_id;
} fn__pq_elem_t;

/*============================================================================
 * Priority Queue Structure
 *============================================================================*/

typedef struct fn__pqueue_s {
  fn__pq_elem_t* elements; /**< Heap array */
  uint32_t count;          /**< Current number of elements */
  uint32_t capacity;       /**< Maximum capacity */
  int is_min_heap;         /**< 1 for min-heap, 0 for max-heap */
} fn__pqueue_t;

/*============================================================================
 * Priority Queue Functions
 *============================================================================*/

fn__pqueue_t* fn__pqueue_create(uint32_t capacity, int is_min_heap);
void fn__pqueue_destroy(fn__pqueue_t* pq);
void fn__pqueue_clear(fn__pqueue_t* pq);
int fn__pqueue_is_empty(const fn__pqueue_t* pq);
int fn__pqueue_is_full(const fn__pqueue_t* pq);
uint32_t fn__pqueue_size(const fn__pqueue_t* pq);
int fn__pqueue_push(fn__pqueue_t* pq, float distance, uint32_t node_id);
int fn__pqueue_pop(fn__pqueue_t* pq, float* out_distance, uint32_t* out_node_id);
int fn__pqueue_peek(const fn__pqueue_t* pq, float* out_distance, uint32_t* out_node_id);
float fn__pqueue_top_distance(const fn__pqueue_t* pq);
int fn__pqueue_try_push(fn__pqueue_t* pq, float distance, uint32_t node_id);
void fn__pqueue_extract_all(fn__pqueue_t* pq, float* distances, uint32_t* node_ids);

#ifdef __cplusplus
}
#endif

#endif /* FLATNAV_INTERNAL_STRUCTURES_H */
