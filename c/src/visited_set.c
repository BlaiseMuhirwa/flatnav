/**
 * @file visited_set.c
 * @brief Visited set implementation for graph traversal
 *
 * Uses a byte-marking scheme that avoids memset on every reset.
 * The marker is incremented each time the set is reset, and nodes
 * are considered visited if their marker equals the current marker.
 */

#include <stdlib.h>
#include <string.h>
#include "internal/structures.h"

/*============================================================================
 * Visited Set Functions
 *============================================================================*/

fn__visited_set_t* fn__visited_set_create(uint32_t capacity) {
  fn__visited_set_t* vs = (fn__visited_set_t*)malloc(sizeof(fn__visited_set_t));
  if (vs == NULL) {
    return NULL;
  }

  vs->markers = (uint8_t*)calloc(capacity, sizeof(uint8_t));
  if (vs->markers == NULL) {
    free(vs);
    return NULL;
  }

  vs->capacity = capacity;
  vs->current_marker = 1;

  return vs;
}

void fn__visited_set_destroy(fn__visited_set_t* vs) {
  if (vs == NULL) {
    return;
  }
  free(vs->markers);
  free(vs);
}

void fn__visited_set_reset(fn__visited_set_t* vs) {
  vs->current_marker++;
  if (vs->current_marker == 0) {
    /* Marker wrapped, need to clear the array */
    memset(vs->markers, 0, vs->capacity * sizeof(uint8_t));
    vs->current_marker = 1;
  }
}

int fn__visited_set_check_and_set(fn__visited_set_t* vs, uint32_t node_id) {
  if (node_id >= vs->capacity) {
    return 0;
  }

  if (vs->markers[node_id] == vs->current_marker) {
    return 1; /* Already visited */
  }

  vs->markers[node_id] = vs->current_marker;
  return 0; /* Not previously visited */
}

int fn__visited_set_is_visited(const fn__visited_set_t* vs, uint32_t node_id) {
  if (node_id >= vs->capacity) {
    return 0;
  }
  return vs->markers[node_id] == vs->current_marker;
}

/*============================================================================
 * Visited Set Pool (Thread-Safe)
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

typedef struct fn__visited_pool_s {
  fn__visited_set_t** sets; /**< Array of visited sets */
  uint32_t count;           /**< Number of sets in pool */
  uint32_t capacity;        /**< Number of nodes per set */
  uint8_t* in_use;          /**< Which sets are currently in use */
  fn__mutex_t lock;         /**< Pool lock */
} fn__visited_pool_t;

fn__visited_pool_t* fn__visited_pool_create(uint32_t node_capacity, uint32_t pool_size) {
  fn__visited_pool_t* pool = (fn__visited_pool_t*)malloc(sizeof(fn__visited_pool_t));
  if (pool == NULL) {
    return NULL;
  }

  pool->sets = (fn__visited_set_t**)calloc(pool_size, sizeof(fn__visited_set_t*));
  pool->in_use = (uint8_t*)calloc(pool_size, sizeof(uint8_t));

  if (pool->sets == NULL || pool->in_use == NULL) {
    free(pool->sets);
    free(pool->in_use);
    free(pool);
    return NULL;
  }

  pool->count = pool_size;
  pool->capacity = node_capacity;
  fn__mutex_init(&pool->lock);

  /* Pre-create all sets */
  for (uint32_t i = 0; i < pool_size; i++) {
    pool->sets[i] = fn__visited_set_create(node_capacity);
    if (pool->sets[i] == NULL) {
      /* Cleanup on failure */
      for (uint32_t j = 0; j < i; j++) {
        fn__visited_set_destroy(pool->sets[j]);
      }
      fn__mutex_destroy(&pool->lock);
      free(pool->sets);
      free(pool->in_use);
      free(pool);
      return NULL;
    }
  }

  return pool;
}

void fn__visited_pool_destroy(fn__visited_pool_t* pool) {
  if (pool == NULL) {
    return;
  }

  for (uint32_t i = 0; i < pool->count; i++) {
    fn__visited_set_destroy(pool->sets[i]);
  }

  fn__mutex_destroy(&pool->lock);
  free(pool->sets);
  free(pool->in_use);
  free(pool);
}

fn__visited_set_t* fn__visited_pool_acquire(fn__visited_pool_t* pool) {
  fn__mutex_lock(&pool->lock);

  for (uint32_t i = 0; i < pool->count; i++) {
    if (!pool->in_use[i]) {
      pool->in_use[i] = 1;
      fn__visited_set_reset(pool->sets[i]);
      fn__mutex_unlock(&pool->lock);
      return pool->sets[i];
    }
  }

  /* All sets in use, create a new one */
  fn__visited_set_t* vs = fn__visited_set_create(pool->capacity);
  fn__mutex_unlock(&pool->lock);
  return vs;
}

void fn__visited_pool_release(fn__visited_pool_t* pool, fn__visited_set_t* vs) {
  fn__mutex_lock(&pool->lock);

  for (uint32_t i = 0; i < pool->count; i++) {
    if (pool->sets[i] == vs) {
      pool->in_use[i] = 0;
      fn__mutex_unlock(&pool->lock);
      return;
    }
  }

  /* Not from pool, was dynamically created */
  fn__visited_set_destroy(vs);
  fn__mutex_unlock(&pool->lock);
}

#else /* FN_NO_THREADS - single-threaded version */

typedef struct fn__visited_pool_s {
  fn__visited_set_t* set; /**< Single visited set */
  uint32_t capacity;      /**< Number of nodes */
} fn__visited_pool_t;

fn__visited_pool_t* fn__visited_pool_create(uint32_t node_capacity, uint32_t pool_size) {
  (void)pool_size;

  fn__visited_pool_t* pool = (fn__visited_pool_t*)malloc(sizeof(fn__visited_pool_t));
  if (pool == NULL) {
    return NULL;
  }

  pool->set = fn__visited_set_create(node_capacity);
  if (pool->set == NULL) {
    free(pool);
    return NULL;
  }

  pool->capacity = node_capacity;
  return pool;
}

void fn__visited_pool_destroy(fn__visited_pool_t* pool) {
  if (pool == NULL) {
    return;
  }
  fn__visited_set_destroy(pool->set);
  free(pool);
}

fn__visited_set_t* fn__visited_pool_acquire(fn__visited_pool_t* pool) {
  fn__visited_set_reset(pool->set);
  return pool->set;
}

void fn__visited_pool_release(fn__visited_pool_t* pool, fn__visited_set_t* vs) {
  (void)pool;
  (void)vs;
  /* No-op for single-threaded version */
}

#endif /* FN_NO_THREADS */
