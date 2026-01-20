/**
 * @file npy_reader.c
 * @brief Simple NPY file reader for float32 and int32 arrays
 */

#include <flatnav/flatnav_npy.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*============================================================================
 * NPY Format Parsing
 *============================================================================*/

/* Internal definition of opaque type */
struct fn_npy_array_s {
  void* data;
  uint32_t shape[2];
  uint32_t ndim;
  int is_float; /* 1 for float32, 0 for int32 */
};

static int fn__parse_npy_header(FILE* fp, fn_npy_array_t* arr) {
  /* Read magic number */
  uint8_t magic[6];
  if (fread(magic, 1, 6, fp) != 6) {
    return -1;
  }

  if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' || magic[3] != 'M' || magic[4] != 'P' ||
      magic[5] != 'Y') {
    return -1; /* Not a valid NPY file */
  }

  /* Read version */
  uint8_t version[2];
  if (fread(version, 1, 2, fp) != 2) {
    return -1;
  }

  /* Read header length */
  uint32_t header_len;
  if (version[0] == 1) {
    uint16_t len16;
    if (fread(&len16, 2, 1, fp) != 1) {
      return -1;
    }
    header_len = len16;
  } else {
    if (fread(&header_len, 4, 1, fp) != 1) {
      return -1;
    }
  }

  /* Read header string */
  char* header = (char*)malloc(header_len + 1);
  if (header == NULL) {
    return -1;
  }
  if (fread(header, 1, header_len, fp) != header_len) {
    free(header);
    return -1;
  }
  header[header_len] = '\0';

  /* Parse dtype */
  arr->is_float = 1; /* Default to float */
  if (strstr(header, "'<i4'") || strstr(header, "'<i8'") || strstr(header, "'int32'") ||
      strstr(header, "'int64'")) {
    arr->is_float = 0;
  }

  /* Parse shape - look for (N,) or (N, M) pattern */
  char* shape_start = strstr(header, "'shape':");
  if (shape_start == NULL) {
    shape_start = strstr(header, "\"shape\":");
  }
  if (shape_start == NULL) {
    free(header);
    return -1;
  }

  /* Find opening paren */
  char* paren = strchr(shape_start, '(');
  if (paren == NULL) {
    free(header);
    return -1;
  }

  /* Parse dimensions */
  arr->ndim = 0;
  arr->shape[0] = 0;
  arr->shape[1] = 0;

  char* p = paren + 1;
  while (*p && *p != ')') {
    if (*p >= '0' && *p <= '9') {
      uint32_t num = 0;
      while (*p >= '0' && *p <= '9') {
        num = num * 10 + (*p - '0');
        p++;
      }
      if (arr->ndim < 2) {
        arr->shape[arr->ndim] = num;
        arr->ndim++;
      }
    } else {
      p++;
    }
  }

  free(header);

  /* Handle 1D arrays */
  if (arr->ndim == 1) {
    arr->shape[1] = 1;
  }

  return 0;
}

/*============================================================================
 * Public API
 *============================================================================*/

FN_API fn_npy_array_t* fn_npy_load(const char* filename) {
  FILE* fp = fopen(filename, "rb");
  if (fp == NULL) {
    return NULL;
  }

  fn_npy_array_t* arr = (fn_npy_array_t*)calloc(1, sizeof(fn_npy_array_t));
  if (arr == NULL) {
    fclose(fp);
    return NULL;
  }

  if (fn__parse_npy_header(fp, arr) != 0) {
    free(arr);
    fclose(fp);
    return NULL;
  }

  /* Calculate data size */
  size_t num_elements = (size_t)arr->shape[0] * arr->shape[1];
  size_t element_size = arr->is_float ? sizeof(float) : sizeof(int32_t);
  size_t data_size = num_elements * element_size;

  /* Allocate and read data */
  arr->data = malloc(data_size);
  if (arr->data == NULL) {
    free(arr);
    fclose(fp);
    return NULL;
  }

  if (fread(arr->data, 1, data_size, fp) != data_size) {
    free(arr->data);
    free(arr);
    fclose(fp);
    return NULL;
  }

  fclose(fp);
  return arr;
}

FN_API void fn_npy_free(fn_npy_array_t* arr) {
  if (arr) {
    free(arr->data);
    free(arr);
  }
}

FN_API float* fn_npy_data_float(fn_npy_array_t* arr) {
  return arr->is_float ? (float*)arr->data : NULL;
}

FN_API int32_t* fn_npy_data_int(fn_npy_array_t* arr) {
  return arr->is_float ? NULL : (int32_t*)arr->data;
}

FN_API uint32_t fn_npy_rows(fn_npy_array_t* arr) {
  return arr->shape[0];
}

FN_API uint32_t fn_npy_cols(fn_npy_array_t* arr) {
  return arr->shape[1];
}
