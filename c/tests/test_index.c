/**
 * @file test_index.c
 * @brief Unit tests for index operations
 */

#include <flatnav/flatnav.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_ASSERT(cond, msg)                                       \
  do {                                                               \
    if (!(cond)) {                                                   \
      fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
      return 1;                                                      \
    }                                                                \
  } while (0)

/*============================================================================
 * Index Creation Tests
 *============================================================================*/

static int test_index_create_destroy(void) {
  printf("  Testing index create/destroy...\n");

  fn_index_config_t config = {.dimension = 32,
                              .max_node_count = 1000,
                              .max_edges_per_node = 16,
                              .metric = FN_METRIC_L2,
                              .data_type = FN_DATA_FLOAT32,
                              .collect_stats = 0};

  fn_index_t* index = NULL;
  fn_error_t err = fn_index_create(&index, &config);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to create index");
  TEST_ASSERT(index != NULL, "Index handle is NULL");

  TEST_ASSERT(fn_index_get_dimension(index) == 32, "Dimension mismatch");
  TEST_ASSERT(fn_index_get_max_nodes(index) == 1000, "Max nodes mismatch");
  TEST_ASSERT(fn_index_get_num_nodes(index) == 0, "Initial node count should be 0");
  TEST_ASSERT(fn_index_get_metric(index) == FN_METRIC_L2, "Metric mismatch");
  TEST_ASSERT(fn_index_get_data_type(index) == FN_DATA_FLOAT32, "Data type mismatch");

  fn_index_destroy(index);
  printf("    PASS\n");
  return 0;
}

static int test_index_create_invalid(void) {
  printf("  Testing index create with invalid params...\n");

  fn_index_config_t config = {0};

  fn_index_t* index = NULL;
  fn_error_t err;

  /* NULL output */
  err = fn_index_create(NULL, &config);
  TEST_ASSERT(err == FN_ERR_NULL_PTR, "Should fail with NULL output");

  /* NULL config */
  err = fn_index_create(&index, NULL);
  TEST_ASSERT(err == FN_ERR_NULL_PTR, "Should fail with NULL config");

  /* Zero dimension */
  config.dimension = 0;
  config.max_node_count = 100;
  config.max_edges_per_node = 16;
  err = fn_index_create(&index, &config);
  TEST_ASSERT(err == FN_ERR_INVALID_ARG, "Should fail with zero dimension");

  printf("    PASS\n");
  return 0;
}

/*============================================================================
 * Add and Search Tests
 *============================================================================*/

static int test_index_add_single(void) {
  printf("  Testing single vector add...\n");

  fn_index_config_t config = {.dimension = 4,
                              .max_node_count = 100,
                              .max_edges_per_node = 8,
                              .metric = FN_METRIC_L2,
                              .data_type = FN_DATA_FLOAT32,
                              .collect_stats = 0};

  fn_index_t* index = NULL;
  fn_error_t err = fn_index_create(&index, &config);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to create index");

  float vec[] = {1.0f, 2.0f, 3.0f, 4.0f};
  err = fn_index_add(index, vec, 42, 16, 1);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to add vector");
  TEST_ASSERT(fn_index_get_num_nodes(index) == 1, "Node count should be 1");

  fn_index_destroy(index);
  printf("    PASS\n");
  return 0;
}

static int test_index_search_basic(void) {
  printf("  Testing basic search...\n");

  fn_index_config_t config = {.dimension = 4,
                              .max_node_count = 100,
                              .max_edges_per_node = 8,
                              .metric = FN_METRIC_L2,
                              .data_type = FN_DATA_FLOAT32,
                              .collect_stats = 0};

  fn_index_t* index = NULL;
  fn_error_t err = fn_index_create(&index, &config);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to create index");

  /* Add some vectors */
  float vecs[][4] = {{0.0f, 0.0f, 0.0f, 0.0f},
                     {1.0f, 0.0f, 0.0f, 0.0f},
                     {0.0f, 1.0f, 0.0f, 0.0f},
                     {0.0f, 0.0f, 1.0f, 0.0f},
                     {1.0f, 1.0f, 1.0f, 1.0f}};

  for (int i = 0; i < 5; i++) {
    err = fn_index_add(index, vecs[i], i, 16, 1);
    TEST_ASSERT(err == FN_ERR_OK, "Failed to add vector");
  }

  TEST_ASSERT(fn_index_get_num_nodes(index) == 5, "Node count should be 5");

  /* Search for the first vector (should find itself) */
  fn_search_params_t params = {.k = 3, .ef_search = 16, .num_initializations = 1};

  fn_result_t results[3];
  err = fn_index_search(index, vecs[0], results, &params);
  TEST_ASSERT(err == FN_ERR_OK, "Search failed");

  /* First result should be the query itself with distance 0 */
  TEST_ASSERT(results[0].label == 0, "Nearest neighbor should be label 0");
  TEST_ASSERT(fabsf(results[0].distance) < 1e-5f, "Distance to self should be ~0");

  fn_index_destroy(index);
  printf("    PASS\n");
  return 0;
}

static int test_index_search_larger(void) {
  printf("  Testing search with more vectors...\n");

  fn_index_config_t config = {.dimension = 32,
                              .max_node_count = 1000,
                              .max_edges_per_node = 16,
                              .metric = FN_METRIC_L2,
                              .data_type = FN_DATA_FLOAT32,
                              .collect_stats = 0};

  fn_index_t* index = NULL;
  fn_error_t err = fn_index_create(&index, &config);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to create index");

  /* Add 100 random vectors */
  for (int i = 0; i < 100; i++) {
    float vec[32];
    for (int j = 0; j < 32; j++) {
      vec[j] = (float)(rand() % 1000) / 1000.0f;
    }
    err = fn_index_add(index, vec, i, 32, 10);
    TEST_ASSERT(err == FN_ERR_OK, "Failed to add vector");
  }

  TEST_ASSERT(fn_index_get_num_nodes(index) == 100, "Node count should be 100");

  /* Search */
  float query[32];
  for (int j = 0; j < 32; j++) {
    query[j] = (float)(rand() % 1000) / 1000.0f;
  }

  fn_search_params_t params = {.k = 10, .ef_search = 50, .num_initializations = 10};

  fn_result_t results[10];
  err = fn_index_search(index, query, results, &params);
  TEST_ASSERT(err == FN_ERR_OK, "Search failed");

  /* Results should be sorted by distance (ascending) */
  for (int i = 1; i < 10; i++) {
    TEST_ASSERT(results[i].distance >= results[i - 1].distance, "Results should be sorted by distance");
  }

  fn_index_destroy(index);
  printf("    PASS\n");
  return 0;
}

/*============================================================================
 * Main
 *============================================================================*/

int main(void) {
  printf("Index Tests\n");
  printf("===========\n");

  fn_init();

  int failures = 0;

  failures += test_index_create_destroy();
  failures += test_index_create_invalid();
  failures += test_index_add_single();
  failures += test_index_search_basic();
  failures += test_index_search_larger();

  fn_cleanup();

  printf("\n");
  if (failures == 0) {
    printf("All tests PASSED\n");
    return 0;
  } else {
    printf("%d test(s) FAILED\n", failures);
    return 1;
  }
}
