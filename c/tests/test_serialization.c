/**
 * @file test_serialization.c
 * @brief Unit tests for serialization
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
 * Serialization Tests
 *============================================================================*/

static int test_save_load_file(void) {
  printf("  Testing save/load to file...\n");

  /* Create and populate index */
  fn_index_config_t config = {.dimension = 8,
                              .max_node_count = 100,
                              .max_edges_per_node = 8,
                              .metric = FN_METRIC_L2,
                              .data_type = FN_DATA_FLOAT32,
                              .collect_stats = 0};

  fn_index_t* index = NULL;
  fn_error_t err = fn_index_create(&index, &config);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to create index");

  /* Add vectors */
  for (int i = 0; i < 50; i++) {
    float vec[8];
    for (int j = 0; j < 8; j++) {
      vec[j] = (float)(i * 8 + j);
    }
    err = fn_index_add(index, vec, i * 10, 16, 1);
    TEST_ASSERT(err == FN_ERR_OK, "Failed to add vector");
  }

  TEST_ASSERT(fn_index_get_num_nodes(index) == 50, "Node count should be 50");

  /* Save to file */
  const char* filename = "/tmp/flatnav_test_index.bin";
  err = fn_index_save(index, filename);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to save index");

  /* Load from file */
  fn_index_t* loaded = NULL;
  err = fn_index_load(&loaded, filename);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to load index");
  TEST_ASSERT(loaded != NULL, "Loaded index is NULL");

  /* Verify properties */
  TEST_ASSERT(fn_index_get_dimension(loaded) == 8, "Dimension mismatch after load");
  TEST_ASSERT(fn_index_get_max_nodes(loaded) == 100, "Max nodes mismatch after load");
  TEST_ASSERT(fn_index_get_num_nodes(loaded) == 50, "Node count mismatch after load");
  TEST_ASSERT(fn_index_get_metric(loaded) == FN_METRIC_L2, "Metric mismatch after load");

  /* Verify search works on loaded index */
  float query[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  fn_search_params_t params = {.k = 5, .ef_search = 20, .num_initializations = 1};
  fn_result_t results[5];

  err = fn_index_search(loaded, query, results, &params);
  TEST_ASSERT(err == FN_ERR_OK, "Search on loaded index failed");

  /* First result should be label 0 (vector 0,1,2,3,4,5,6,7) */
  TEST_ASSERT(results[0].label == 0, "Expected label 0 as nearest neighbor");

  fn_index_destroy(index);
  fn_index_destroy(loaded);

  /* Clean up test file */
  remove(filename);

  printf("    PASS\n");
  return 0;
}

static int test_buffer_serialization(void) {
  printf("  Testing buffer serialization...\n");

  /* Create and populate index */
  fn_index_config_t config = {.dimension = 4,
                              .max_node_count = 20,
                              .max_edges_per_node = 4,
                              .metric = FN_METRIC_IP,
                              .data_type = FN_DATA_FLOAT32,
                              .collect_stats = 0};

  fn_index_t* index = NULL;
  fn_error_t err = fn_index_create(&index, &config);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to create index");

  /* Add vectors */
  for (int i = 0; i < 10; i++) {
    float vec[4] = {(float)i, (float)(i + 1), (float)(i + 2), (float)(i + 3)};
    err = fn_index_add(index, vec, i, 8, 1);
    TEST_ASSERT(err == FN_ERR_OK, "Failed to add vector");
  }

  /* Get required buffer size */
  size_t buffer_size = fn_index_serialized_size(index);
  TEST_ASSERT(buffer_size > 0, "Serialized size should be > 0");

  /* Allocate buffer */
  void* buffer = malloc(buffer_size);
  TEST_ASSERT(buffer != NULL, "Failed to allocate buffer");

  /* Serialize to buffer */
  size_t actual_size = buffer_size;
  err = fn_index_to_buffer(index, buffer, &actual_size);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to serialize to buffer");
  TEST_ASSERT(actual_size == buffer_size, "Actual size should match expected");

  /* Deserialize from buffer */
  fn_index_t* loaded = NULL;
  err = fn_index_from_buffer(&loaded, buffer, buffer_size);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to deserialize from buffer");
  TEST_ASSERT(loaded != NULL, "Loaded index is NULL");

  /* Verify properties */
  TEST_ASSERT(fn_index_get_dimension(loaded) == 4, "Dimension mismatch");
  TEST_ASSERT(fn_index_get_num_nodes(loaded) == 10, "Node count mismatch");
  TEST_ASSERT(fn_index_get_metric(loaded) == FN_METRIC_IP, "Metric mismatch");

  free(buffer);
  fn_index_destroy(index);
  fn_index_destroy(loaded);

  printf("    PASS\n");
  return 0;
}

static int test_invalid_format(void) {
  printf("  Testing invalid format handling...\n");

  /* Try to load non-existent file */
  fn_index_t* index = NULL;
  fn_error_t err = fn_index_load(&index, "/nonexistent/path/file.bin");
  TEST_ASSERT(err == FN_ERR_IO_ERROR, "Should fail with IO error for missing file");

  /* Try to load from invalid buffer */
  char bad_buffer[] = "not a valid flatnav index";
  err = fn_index_from_buffer(&index, bad_buffer, sizeof(bad_buffer));
  TEST_ASSERT(err == FN_ERR_INVALID_FORMAT, "Should fail with invalid format");

  /* Try with buffer too small */
  char small_buffer[10] = {0};
  err = fn_index_from_buffer(&index, small_buffer, sizeof(small_buffer));
  TEST_ASSERT(err == FN_ERR_INVALID_FORMAT, "Should fail with small buffer");

  printf("    PASS\n");
  return 0;
}

/*============================================================================
 * Main
 *============================================================================*/

int main(void) {
  printf("Serialization Tests\n");
  printf("===================\n");

  fn_init();

  int failures = 0;

  failures += test_save_load_file();
  failures += test_buffer_serialization();
  failures += test_invalid_format();

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
