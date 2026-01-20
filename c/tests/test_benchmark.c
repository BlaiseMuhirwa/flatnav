/**
 * @file test_benchmark.c
 * @brief Benchmark test using real embedding data (SIFT-128)
 *
 * Tests the C API against the SIFT-128 Euclidean dataset and measures recall.
 */

#include <flatnav/flatnav.h>
#include <flatnav/flatnav_npy.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef FN_NO_THREADS
#include <pthread.h>
#endif

/*============================================================================
 * Configuration
 *============================================================================*/

/* Full dataset configuration */
#define MAX_TRAIN_VECTORS 1000000 /* Use full 1M vectors */
#define MAX_TEST_QUERIES 10000    /* Use all 10K queries */
#define K 10                      /* Number of neighbors to retrieve */
#define M 32                      /* Max edges per node */
#define EF_CONSTRUCTION 200       /* Construction ef */
#define EF_SEARCH 200             /* Search ef */
#define NUM_INIT_CONSTRUCT 100    /* Entry point samples during construction */
#define NUM_THREADS 8             /* Number of threads for parallel indexing */

/* Data paths - relative to c/build directory */
static const char* TRAIN_PATH = "../../data/sift-128-euclidean/sift-128-euclidean.train.npy";
static const char* TEST_PATH = "../../data/sift-128-euclidean/sift-128-euclidean.test.npy";
static const char* GTRUTH_PATH = "../../data/sift-128-euclidean/sift-128-euclidean.gtruth.npy";

/*============================================================================
 * Parallel Indexing Support
 *============================================================================*/

#ifndef FN_NO_THREADS
typedef struct {
  fn_index_t* index;
  float* train_data;
  uint32_t dim;
  uint32_t n_train;
  atomic_uint* current;
  atomic_int* error_flag;
} parallel_add_ctx_t;

static void* parallel_add_worker(void* arg) {
  parallel_add_ctx_t* ctx = (parallel_add_ctx_t*)arg;
  while (1) {
    uint32_t i = atomic_fetch_add(ctx->current, 1);
    if (i >= ctx->n_train)
      break;

    float* vec = ctx->train_data + i * ctx->dim;
    fn_error_t local_err = fn_index_add(ctx->index, vec, (fn_label_t)i, EF_CONSTRUCTION, NUM_INIT_CONSTRUCT);
    if (local_err != FN_ERR_OK) {
      atomic_store(ctx->error_flag, 1);
    }
  }
  return NULL;
}
#endif

/*============================================================================
 * Recall Calculation
 *============================================================================*/

static double compute_recall(const fn_result_t* results, const int32_t* ground_truth, uint32_t k,
                             uint32_t gt_k) {
  int hits = 0;
  for (uint32_t i = 0; i < k; i++) {
    int32_t result_label = results[i].label;
    for (uint32_t j = 0; j < k && j < gt_k; j++) {
      if (result_label == ground_truth[j]) {
        hits++;
        break;
      }
    }
  }
  return (double)hits / k;
}

/*============================================================================
 * Main Benchmark
 *============================================================================*/

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  printf("FlatNav C API Benchmark\n");
  printf("=======================\n\n");

  fn_init();

  /* Load training data */
  printf("Loading training data...\n");
  fn_npy_array_t* train_arr = fn_npy_load(TRAIN_PATH);
  if (train_arr == NULL) {
    fprintf(stderr, "ERROR: Failed to load training data from %s\n", TRAIN_PATH);
    fprintf(stderr, "Make sure to run this test from the c/build directory\n");
    return 1;
  }

  float* train_data = fn_npy_data_float(train_arr);
  uint32_t n_train = fn_npy_rows(train_arr);
  uint32_t dim = fn_npy_cols(train_arr);

  printf("  Loaded %u vectors of dimension %u\n", n_train, dim);

  /* Limit training data size */
  if (n_train > MAX_TRAIN_VECTORS) {
    n_train = MAX_TRAIN_VECTORS;
    printf("  Using first %u vectors for indexing\n", n_train);
  }

  /* Load test queries */
  printf("Loading test queries...\n");
  fn_npy_array_t* test_arr = fn_npy_load(TEST_PATH);
  if (test_arr == NULL) {
    fprintf(stderr, "ERROR: Failed to load test data from %s\n", TEST_PATH);
    fn_npy_free(train_arr);
    return 1;
  }

  float* test_data = fn_npy_data_float(test_arr);
  uint32_t n_test = fn_npy_rows(test_arr);
  uint32_t test_dim = fn_npy_cols(test_arr);

  if (test_dim != dim) {
    fprintf(stderr, "ERROR: Dimension mismatch between train and test data\n");
    fn_npy_free(train_arr);
    fn_npy_free(test_arr);
    return 1;
  }

  printf("  Loaded %u queries\n", n_test);

  /* Limit test queries */
  if (n_test > MAX_TEST_QUERIES) {
    n_test = MAX_TEST_QUERIES;
    printf("  Using first %u queries\n", n_test);
  }

  /* Load ground truth */
  printf("Loading ground truth...\n");
  fn_npy_array_t* gtruth_arr = fn_npy_load(GTRUTH_PATH);
  if (gtruth_arr == NULL) {
    fprintf(stderr, "ERROR: Failed to load ground truth from %s\n", GTRUTH_PATH);
    fn_npy_free(train_arr);
    fn_npy_free(test_arr);
    return 1;
  }

  int32_t* gtruth_data = fn_npy_data_int(gtruth_arr);
  uint32_t gt_k = fn_npy_cols(gtruth_arr);

  printf("  Loaded ground truth with %u neighbors per query\n", gt_k);

  /* Create index */
  printf("\nBuilding index...\n");
  printf("  M=%d, ef_construction=%d\n", M, EF_CONSTRUCTION);

  fn_index_config_t config = {.dimension = dim,
                              .max_node_count = n_train,
                              .max_edges_per_node = M,
                              .metric = FN_METRIC_L2,
                              .data_type = FN_DATA_FLOAT32,
                              .collect_stats = 0};

  fn_index_t* index = NULL;
  fn_error_t err = fn_index_create(&index, &config);
  if (err != FN_ERR_OK) {
    fprintf(stderr, "ERROR: Failed to create index: %s\n", fn_error_string(err));
    fn_npy_free(train_arr);
    fn_npy_free(test_arr);
    fn_npy_free(gtruth_arr);
    return 1;
  }

  /* Add vectors to index */
#ifndef FN_NO_THREADS
  /* Parallel indexing using pthreads (similar to C++ Multithreading.h) */
  printf("  Using %d threads for parallel indexing\n", NUM_THREADS);

  atomic_uint current = 0;
  atomic_int error_flag = 0;

  parallel_add_ctx_t ctx = {.index = index,
                            .train_data = train_data,
                            .dim = dim,
                            .n_train = n_train,
                            .current = &current,
                            .error_flag = &error_flag};

  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);

  pthread_t threads[NUM_THREADS];
  for (int t = 0; t < NUM_THREADS; t++) {
    pthread_create(&threads[t], NULL, parallel_add_worker, &ctx);
  }
  for (int t = 0; t < NUM_THREADS; t++) {
    pthread_join(threads[t], NULL);
  }

  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  double build_time_ms =
      (ts_end.tv_sec - ts_start.tv_sec) * 1000.0 + (ts_end.tv_nsec - ts_start.tv_nsec) / 1000000.0;

  if (atomic_load(&error_flag)) {
    fprintf(stderr, "ERROR: Failed during parallel indexing\n");
    fn_index_destroy(index);
    fn_npy_free(train_arr);
    fn_npy_free(test_arr);
    fn_npy_free(gtruth_arr);
    return 1;
  }
#else
  clock_t build_start = clock();

  for (uint32_t i = 0; i < n_train; i++) {
    float* vec = train_data + i * dim;
    err = fn_index_add(index, vec, (fn_label_t)i, EF_CONSTRUCTION, NUM_INIT_CONSTRUCT);
    if (err != FN_ERR_OK) {
      fprintf(stderr, "ERROR: Failed to add vector %u: %s\n", i, fn_error_string(err));
      fn_index_destroy(index);
      fn_npy_free(train_arr);
      fn_npy_free(test_arr);
      fn_npy_free(gtruth_arr);
      return 1;
    }

    if ((i + 1) % 100000 == 0 || i + 1 == n_train) {
      printf("\r  Added %u/%u vectors...", i + 1, n_train);
      fflush(stdout);
    }
  }

  clock_t build_end = clock();
  double build_time_ms = (double)(build_end - build_start) * 1000.0 / CLOCKS_PER_SEC;
#endif

  printf("\n  Build time: %.1f ms (%.1f vectors/sec)\n", build_time_ms, n_train / (build_time_ms / 1000.0));

  /* Direct distance test - verify distance computation works */
  printf("\nDirect distance test...\n");
  {
    const fn_distance_t* dist = fn_index_get_distance(index);
    float* vec0 = train_data;
    float* vec1 = train_data + dim;

    /* Distance from vec0 to itself should be 0 */
    float self_dist = fn_distance_compute(dist, vec0, vec0);
    printf("  Distance vec0 to vec0: %.6f (should be 0)\n", self_dist);

    /* Distance from vec0 to vec1 */
    float cross_dist = fn_distance_compute(dist, vec0, vec1);
    printf("  Distance vec0 to vec1: %.6f\n", cross_dist);

    /* Test: search with query = exactly train_data[0], should return 0 with dist 0 */
    fn_search_params_t test_params = {.k = 1, .ef_search = 200, .num_initializations = 1};
    fn_result_t test_result;
    err = fn_index_search(index, vec0, &test_result, &test_params);
    printf("  Search for vec0: label=%d, dist=%.6f (expected label=0, dist=0)\n", test_result.label,
           test_result.distance);

    /* Try with higher num_init */
    test_params.num_initializations = 1000;
    err = fn_index_search(index, vec1, &test_result, &test_params);
    printf("  Search for vec1 (1000 inits): label=%d, dist=%.6f (expected label=1, dist=0)\n",
           test_result.label, test_result.distance);

    /* Test: search for different vectors with consistent num_initializations */
    printf("\n  Testing search consistency...\n");
    for (int target = 0; target < 5; target++) {
      test_params.num_initializations = 100;
      err = fn_index_search(index, train_data + target * dim, &test_result, &test_params);
      printf("    Search train_data[%d] (100 inits): label=%d, dist=%.2f\n", target, test_result.label,
             test_result.distance);
    }
  }

  /* Sanity check: search for indexed vectors and verify they find themselves */
  printf("\nRunning sanity check (self-search)...\n");
  {
    fn_search_params_t sanity_params = {.k = 1, .ef_search = EF_SEARCH, .num_initializations = 100};

    fn_result_t sanity_result;
    int self_found = 0;
    int num_checks = 100; /* Check first 100 vectors */

    for (int i = 0; i < num_checks && (uint32_t)i < n_train; i++) {
      float* vec = train_data + i * dim;
      err = fn_index_search(index, vec, &sanity_result, &sanity_params);
      if (err == FN_ERR_OK && sanity_result.label == i) {
        self_found++;
      } else if (i < 5) {
        /* Debug: print first few failures */
        printf("    Query %d: found label=%d, dist=%.6f (expected %d)\n", i, sanity_result.label,
               sanity_result.distance, i);
      }
    }

    printf("  Self-search accuracy: %d/%d (%.1f%%)\n", self_found, num_checks,
           100.0 * self_found / num_checks);

    if (self_found < num_checks * 0.9) {
      printf("  WARNING: Self-search accuracy is low, search may be broken!\n");
    }
  }

  /* Search and compute recall */
  printf("\nSearching (k=%d, ef_search=%d)...\n", K, EF_SEARCH);

  fn_search_params_t params = {.k = K, .ef_search = EF_SEARCH, .num_initializations = 100};

  fn_result_t* results = (fn_result_t*)malloc(K * sizeof(fn_result_t));
  if (results == NULL) {
    fprintf(stderr, "ERROR: Out of memory\n");
    fn_index_destroy(index);
    fn_npy_free(train_arr);
    fn_npy_free(test_arr);
    fn_npy_free(gtruth_arr);
    return 1;
  }

  double total_recall = 0.0;
  clock_t search_start = clock();

  for (uint32_t q = 0; q < n_test; q++) {
    float* query = test_data + q * dim;
    int32_t* gt = gtruth_data + q * gt_k;

    err = fn_index_search(index, query, results, &params);
    if (err != FN_ERR_OK) {
      fprintf(stderr, "ERROR: Search failed for query %u: %s\n", q, fn_error_string(err));
      continue;
    }

    /* Note: Ground truth is computed against FULL dataset (1M vectors),
         * but we only indexed n_train vectors. We need to filter ground truth
         * to only include labels < n_train */
    int32_t filtered_gt[K];
    int filtered_count = 0;
    for (uint32_t i = 0; i < gt_k && filtered_count < K; i++) {
      if (gt[i] < (int32_t)n_train) {
        filtered_gt[filtered_count++] = gt[i];
      }
    }

    if (filtered_count > 0) {
      /* Compute recall against filtered ground truth */
      int hits = 0;
      for (int i = 0; i < K; i++) {
        int32_t result_label = results[i].label;
        for (int j = 0; j < filtered_count; j++) {
          if (result_label == filtered_gt[j]) {
            hits++;
            break;
          }
        }
      }
      total_recall += (double)hits / filtered_count;
    }
  }

  clock_t search_end = clock();
  double search_time_ms = (double)(search_end - search_start) * 1000.0 / CLOCKS_PER_SEC;

  double mean_recall = total_recall / n_test;
  double qps = n_test / (search_time_ms / 1000.0);

  printf("  Search time: %.1f ms (%.1f QPS)\n", search_time_ms, qps);
  printf("\n");
  printf("Results\n");
  printf("-------\n");
  printf("  Mean Recall@%d: %.4f (%.1f%%)\n", K, mean_recall, mean_recall * 100.0);

  /* Check recall threshold */
  int success = 1;
  double min_recall = 0.90; /* Expect at least 90% recall */

  if (mean_recall < min_recall) {
    printf("\n  WARNING: Recall %.1f%% is below expected threshold %.1f%%\n", mean_recall * 100.0,
           min_recall * 100.0);
    success = 0;
  } else {
    printf("\n  SUCCESS: Recall meets threshold (>= %.1f%%)\n", min_recall * 100.0);
  }

  /* Cleanup */
  free(results);
  fn_index_destroy(index);
  fn_npy_free(train_arr);
  fn_npy_free(test_arr);
  fn_npy_free(gtruth_arr);
  fn_cleanup();

  return success ? 0 : 1;
}
