/**
 * @file test_distance.c
 * @brief Unit tests for distance functions
 */

#include <flatnav/flatnav.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_ASSERT(cond, msg)                                       \
  do {                                                               \
    if (!(cond)) {                                                   \
      fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
      return 1;                                                      \
    }                                                                \
  } while (0)

#define TEST_ASSERT_NEAR(a, b, eps, msg)                                                               \
  do {                                                                                                 \
    if (fabsf((a) - (b)) > (eps)) {                                                                    \
      fprintf(stderr, "FAIL: %s:%d: %s (expected %f, got %f)\n", __FILE__, __LINE__, msg, (double)(b), \
              (double)(a));                                                                            \
      return 1;                                                                                        \
    }                                                                                                  \
  } while (0)

/*============================================================================
 * L2 Distance Tests
 *============================================================================*/

static int test_l2_float32(void) {
  printf("  Testing L2 float32...\n");

  fn_distance_t* dist = NULL;
  fn_error_t err = fn_distance_create_l2(&dist, 4, FN_DATA_FLOAT32);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to create L2 distance");
  TEST_ASSERT(dist != NULL, "Distance handle is NULL");

  float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float y[] = {5.0f, 6.0f, 7.0f, 8.0f};

  /* Expected: (1-5)^2 + (2-6)^2 + (3-7)^2 + (4-8)^2 = 16 + 16 + 16 + 16 = 64 */
  float d = fn_distance_compute(dist, x, y);
  TEST_ASSERT_NEAR(d, 64.0f, 1e-5f, "L2 distance incorrect");

  /* Test identical vectors */
  d = fn_distance_compute(dist, x, x);
  TEST_ASSERT_NEAR(d, 0.0f, 1e-5f, "L2 distance of identical vectors should be 0");

  fn_distance_destroy(dist);
  printf("    PASS\n");
  return 0;
}

static int test_l2_int8(void) {
  printf("  Testing L2 int8...\n");

  fn_distance_t* dist = NULL;
  fn_error_t err = fn_distance_create_l2(&dist, 4, FN_DATA_INT8);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to create L2 distance");

  int8_t x[] = {1, 2, 3, 4};
  int8_t y[] = {5, 6, 7, 8};

  float d = fn_distance_compute(dist, x, y);
  TEST_ASSERT_NEAR(d, 64.0f, 1e-5f, "L2 distance incorrect");

  fn_distance_destroy(dist);
  printf("    PASS\n");
  return 0;
}

static int test_l2_high_dimension(void) {
  printf("  Testing L2 high dimension (128)...\n");

  fn_distance_t* dist = NULL;
  fn_error_t err = fn_distance_create_l2(&dist, 128, FN_DATA_FLOAT32);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to create L2 distance");

  float* x = (float*)malloc(128 * sizeof(float));
  float* y = (float*)malloc(128 * sizeof(float));

  float expected = 0.0f;
  for (int i = 0; i < 128; i++) {
    x[i] = (float)i;
    y[i] = (float)(i + 1);
    expected += 1.0f; /* Each diff is 1, squared is 1 */
  }

  float d = fn_distance_compute(dist, x, y);
  TEST_ASSERT_NEAR(d, expected, 1e-3f, "L2 distance incorrect for high dim");

  free(x);
  free(y);
  fn_distance_destroy(dist);
  printf("    PASS\n");
  return 0;
}

/*============================================================================
 * IP Distance Tests
 *============================================================================*/

static int test_ip_float32(void) {
  printf("  Testing IP float32...\n");

  fn_distance_t* dist = NULL;
  fn_error_t err = fn_distance_create_ip(&dist, 4, FN_DATA_FLOAT32);
  TEST_ASSERT(err == FN_ERR_OK, "Failed to create IP distance");

  float x[] = {1.0f, 0.0f, 0.0f, 0.0f};
  float y[] = {1.0f, 0.0f, 0.0f, 0.0f};

  /* Dot product = 1, so distance = 1 - 1 = 0 */
  float d = fn_distance_compute(dist, x, y);
  TEST_ASSERT_NEAR(d, 0.0f, 1e-5f, "IP distance of identical unit vectors should be 0");

  /* Orthogonal vectors */
  float z[] = {0.0f, 1.0f, 0.0f, 0.0f};
  d = fn_distance_compute(dist, x, z);
  TEST_ASSERT_NEAR(d, 1.0f, 1e-5f, "IP distance of orthogonal vectors should be 1");

  fn_distance_destroy(dist);
  printf("    PASS\n");
  return 0;
}

/*============================================================================
 * Main
 *============================================================================*/

int main(void) {
  printf("Distance Function Tests\n");
  printf("=======================\n");

  fn_init();

  int failures = 0;

  failures += test_l2_float32();
  failures += test_l2_int8();
  failures += test_l2_high_dimension();
  failures += test_ip_float32();

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
