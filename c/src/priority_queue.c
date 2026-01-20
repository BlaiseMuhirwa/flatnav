/**
 * @file priority_queue.c
 * @brief Binary heap priority queue implementation
 *
 * Provides both min-heap (for candidates) and max-heap (for results)
 * operations for the beam search algorithm.
 */

#include <stdlib.h>
#include <string.h>
#include "internal/structures.h"

/*============================================================================
 * Internal Heap Operations
 *============================================================================*/

static void fn__pq_swap(fn__pq_elem_t* a, fn__pq_elem_t* b) {
  fn__pq_elem_t tmp = *a;
  *a = *b;
  *b = tmp;
}

static int fn__pq_compare(const fn__pqueue_t* pq, uint32_t i, uint32_t j) {
  /* For min-heap: return 1 if elements[i] < elements[j] */
  /* For max-heap: return 1 if elements[i] > elements[j] */
  if (pq->is_min_heap) {
    return pq->elements[i].distance < pq->elements[j].distance;
  } else {
    return pq->elements[i].distance > pq->elements[j].distance;
  }
}

static void fn__pq_sift_up(fn__pqueue_t* pq, uint32_t idx) {
  while (idx > 0) {
    uint32_t parent = (idx - 1) / 2;
    if (fn__pq_compare(pq, idx, parent)) {
      fn__pq_swap(&pq->elements[idx], &pq->elements[parent]);
      idx = parent;
    } else {
      break;
    }
  }
}

static void fn__pq_sift_down(fn__pqueue_t* pq, uint32_t idx) {
  while (1) {
    uint32_t smallest = idx;
    uint32_t left = 2 * idx + 1;
    uint32_t right = 2 * idx + 2;

    if (left < pq->count && fn__pq_compare(pq, left, smallest)) {
      smallest = left;
    }
    if (right < pq->count && fn__pq_compare(pq, right, smallest)) {
      smallest = right;
    }

    if (smallest != idx) {
      fn__pq_swap(&pq->elements[idx], &pq->elements[smallest]);
      idx = smallest;
    } else {
      break;
    }
  }
}

/*============================================================================
 * Priority Queue Public Functions
 *============================================================================*/

fn__pqueue_t* fn__pqueue_create(uint32_t capacity, int is_min_heap) {
  fn__pqueue_t* pq = (fn__pqueue_t*)malloc(sizeof(fn__pqueue_t));
  if (pq == NULL) {
    return NULL;
  }

  pq->elements = (fn__pq_elem_t*)malloc(capacity * sizeof(fn__pq_elem_t));
  if (pq->elements == NULL) {
    free(pq);
    return NULL;
  }

  pq->count = 0;
  pq->capacity = capacity;
  pq->is_min_heap = is_min_heap;

  return pq;
}

void fn__pqueue_destroy(fn__pqueue_t* pq) {
  if (pq == NULL) {
    return;
  }
  free(pq->elements);
  free(pq);
}

void fn__pqueue_clear(fn__pqueue_t* pq) {
  pq->count = 0;
}

int fn__pqueue_is_empty(const fn__pqueue_t* pq) {
  return pq->count == 0;
}

int fn__pqueue_is_full(const fn__pqueue_t* pq) {
  return pq->count >= pq->capacity;
}

uint32_t fn__pqueue_size(const fn__pqueue_t* pq) {
  return pq->count;
}

int fn__pqueue_push(fn__pqueue_t* pq, float distance, uint32_t node_id) {
  if (pq->count >= pq->capacity) {
    return 0; /* Queue is full */
  }

  uint32_t idx = pq->count;
  pq->elements[idx].distance = distance;
  pq->elements[idx].node_id = node_id;
  pq->count++;

  fn__pq_sift_up(pq, idx);
  return 1;
}

int fn__pqueue_pop(fn__pqueue_t* pq, float* out_distance, uint32_t* out_node_id) {
  if (pq->count == 0) {
    return 0; /* Queue is empty */
  }

  if (out_distance) {
    *out_distance = pq->elements[0].distance;
  }
  if (out_node_id) {
    *out_node_id = pq->elements[0].node_id;
  }

  pq->count--;
  if (pq->count > 0) {
    pq->elements[0] = pq->elements[pq->count];
    fn__pq_sift_down(pq, 0);
  }

  return 1;
}

int fn__pqueue_peek(const fn__pqueue_t* pq, float* out_distance, uint32_t* out_node_id) {
  if (pq->count == 0) {
    return 0;
  }

  if (out_distance) {
    *out_distance = pq->elements[0].distance;
  }
  if (out_node_id) {
    *out_node_id = pq->elements[0].node_id;
  }

  return 1;
}

float fn__pqueue_top_distance(const fn__pqueue_t* pq) {
  if (pq->count == 0) {
    return pq->is_min_heap ? 1e30f : -1e30f;
  }
  return pq->elements[0].distance;
}

/**
 * @brief Try to insert, replacing worst element if queue is full and new is better
 *
 * For a max-heap (results), this inserts if distance < top (better result).
 * For a min-heap (candidates), this inserts if distance > top (shouldn't happen normally).
 */
int fn__pqueue_try_push(fn__pqueue_t* pq, float distance, uint32_t node_id) {
  if (pq->count < pq->capacity) {
    return fn__pqueue_push(pq, distance, node_id);
  }

  /* Queue is full - check if we should replace the top */
  if (pq->is_min_heap) {
    /* For min-heap: replace if new distance is larger (shouldn't normally happen) */
    if (distance > pq->elements[0].distance) {
      pq->elements[0].distance = distance;
      pq->elements[0].node_id = node_id;
      fn__pq_sift_down(pq, 0);
      return 1;
    }
  } else {
    /* For max-heap (results): replace if new distance is smaller (better) */
    if (distance < pq->elements[0].distance) {
      pq->elements[0].distance = distance;
      pq->elements[0].node_id = node_id;
      fn__pq_sift_down(pq, 0);
      return 1;
    }
  }

  return 0; /* Not inserted */
}

/**
 * @brief Extract all elements in sorted order (ascending for min-heap, descending for max-heap)
 */
void fn__pqueue_extract_all(fn__pqueue_t* pq, float* distances, uint32_t* node_ids) {
  uint32_t idx = 0;
  while (pq->count > 0) {
    fn__pqueue_pop(pq, &distances[idx], &node_ids[idx]);
    idx++;
  }
}
