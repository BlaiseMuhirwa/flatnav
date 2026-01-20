/**
 * @file serialization.c
 * @brief Binary serialization for FlatNav C API
 *
 * Implements save/load functionality for indices using a custom binary format.
 */

#include <flatnav/flatnav.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*============================================================================
 * Binary Format Constants
 *============================================================================*/

#define FN_MAGIC_0 'F'
#define FN_MAGIC_1 'N'
#define FN_MAGIC_2 'A'
#define FN_MAGIC_3 'V'
#define FN_FORMAT_VERSION 1

/*============================================================================
 * Binary Header Structure
 *============================================================================*/

/*
 * FlatNav Binary Index Format (Version 1)
 *
 * HEADER (64 bytes):
 * +0x00: magic[4]       = "FNAV"
 * +0x04: version        = uint32_t (1)
 * +0x08: data_type      = uint8_t
 * +0x09: metric_type    = uint8_t
 * +0x0A: reserved[2]    = uint8_t[2]
 * +0x0C: dimension      = uint32_t
 * +0x10: max_node_count = uint32_t
 * +0x14: cur_num_nodes  = uint32_t
 * +0x18: M              = uint32_t
 * +0x1C: data_size      = uint32_t
 * +0x20: node_size      = uint32_t
 * +0x24: label_size     = uint32_t
 * +0x28: flags          = uint32_t
 * +0x2C: reserved[20]   = uint8_t[20]
 *
 * NODE DATA:
 * +0x40: node_memory[node_size * max_node_count]
 */

typedef struct fn__header_s {
  char magic[4];
  uint32_t version;
  uint8_t data_type;
  uint8_t metric_type;
  uint8_t reserved1[2];
  uint32_t dimension;
  uint32_t max_node_count;
  uint32_t cur_num_nodes;
  uint32_t M;
  uint32_t data_size_bytes;
  uint32_t node_size_bytes;
  uint32_t label_size;
  uint32_t flags;
  uint8_t reserved2[20];
} fn__header_t;

/* Ensure header is exactly 64 bytes */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(fn__header_t) == 64, "Header must be 64 bytes");
#endif

/*============================================================================
 * Internal Index Access
 *============================================================================*/

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
} fn__index_internal_t;

/*============================================================================
 * Serialization Functions
 *============================================================================*/

FN_API size_t fn_index_serialized_size(const fn_index_t* index) {
  if (index == NULL) {
    return 0;
  }

  const fn__index_internal_t* idx = (const fn__index_internal_t*)index;

  return sizeof(fn__header_t) + (size_t)idx->node_size_bytes * idx->max_node_count;
}

FN_API fn_error_t fn_index_save(const fn_index_t* index, const char* filename) {
  if (index == NULL || filename == NULL) {
    return FN_ERR_NULL_PTR;
  }

  const fn__index_internal_t* idx = (const fn__index_internal_t*)index;

  FILE* f = fopen(filename, "wb");
  if (f == NULL) {
    return FN_ERR_IO_ERROR;
  }

  /* Prepare header */
  fn__header_t header;
  memset(&header, 0, sizeof(header));

  header.magic[0] = FN_MAGIC_0;
  header.magic[1] = FN_MAGIC_1;
  header.magic[2] = FN_MAGIC_2;
  header.magic[3] = FN_MAGIC_3;
  header.version = FN_FORMAT_VERSION;
  header.data_type = (uint8_t)idx->distance->data_type;
  header.metric_type = (uint8_t)idx->distance->metric;
  header.dimension = idx->distance->dimension;
  header.max_node_count = idx->max_node_count;
  header.cur_num_nodes = idx->cur_num_nodes;
  header.M = idx->M;
  header.data_size_bytes = idx->data_size_bytes;
  header.node_size_bytes = idx->node_size_bytes;
  header.label_size = sizeof(fn_label_t);
  header.flags = 0;

  /* Write header */
  if (fwrite(&header, sizeof(header), 1, f) != 1) {
    fclose(f);
    return FN_ERR_IO_ERROR;
  }

  /* Write node memory */
  size_t total_size = (size_t)idx->node_size_bytes * idx->max_node_count;
  if (fwrite(idx->node_memory, 1, total_size, f) != total_size) {
    fclose(f);
    return FN_ERR_IO_ERROR;
  }

  fclose(f);
  return FN_ERR_OK;
}

FN_API fn_error_t fn_index_load(fn_index_t** out_index, const char* filename) {
  if (out_index == NULL || filename == NULL) {
    return FN_ERR_NULL_PTR;
  }

  FILE* f = fopen(filename, "rb");
  if (f == NULL) {
    return FN_ERR_IO_ERROR;
  }

  /* Read header */
  fn__header_t header;
  if (fread(&header, sizeof(header), 1, f) != 1) {
    fclose(f);
    return FN_ERR_IO_ERROR;
  }

  /* Validate magic */
  if (header.magic[0] != FN_MAGIC_0 || header.magic[1] != FN_MAGIC_1 || header.magic[2] != FN_MAGIC_2 ||
      header.magic[3] != FN_MAGIC_3) {
    fclose(f);
    return FN_ERR_INVALID_FORMAT;
  }

  /* Validate version */
  if (header.version != FN_FORMAT_VERSION) {
    fclose(f);
    return FN_ERR_INVALID_FORMAT;
  }

  /* Validate label size */
  if (header.label_size != sizeof(fn_label_t)) {
    fclose(f);
    return FN_ERR_INVALID_FORMAT;
  }

  /* Create index */
  fn_index_config_t config;
  memset(&config, 0, sizeof(config));
  config.dimension = header.dimension;
  config.max_node_count = header.max_node_count;
  config.max_edges_per_node = header.M;
  config.metric = (fn_metric_type_t)header.metric_type;
  config.data_type = (fn_data_type_t)header.data_type;
  config.collect_stats = 0;

  fn_error_t err = fn_index_create(out_index, &config);
  if (err != FN_ERR_OK) {
    fclose(f);
    return err;
  }

  fn__index_internal_t* idx = (fn__index_internal_t*)(*out_index);

  /* Validate node size matches */
  if (header.node_size_bytes != idx->node_size_bytes) {
    fn_index_destroy(*out_index);
    *out_index = NULL;
    fclose(f);
    return FN_ERR_INVALID_FORMAT;
  }

  /* Read node memory */
  size_t total_size = (size_t)header.node_size_bytes * header.max_node_count;
  if (fread(idx->node_memory, 1, total_size, f) != total_size) {
    fn_index_destroy(*out_index);
    *out_index = NULL;
    fclose(f);
    return FN_ERR_IO_ERROR;
  }

  /* Set current node count */
  idx->cur_num_nodes = header.cur_num_nodes;

  fclose(f);
  return FN_ERR_OK;
}

FN_API fn_error_t fn_index_to_buffer(const fn_index_t* index, void* buffer, size_t* buffer_size) {
  if (index == NULL || buffer_size == NULL) {
    return FN_ERR_NULL_PTR;
  }

  const fn__index_internal_t* idx = (const fn__index_internal_t*)index;

  size_t required_size = sizeof(fn__header_t) + (size_t)idx->node_size_bytes * idx->max_node_count;

  if (buffer == NULL) {
    *buffer_size = required_size;
    return FN_ERR_OK;
  }

  if (*buffer_size < required_size) {
    *buffer_size = required_size;
    return FN_ERR_INVALID_ARG;
  }

  char* ptr = (char*)buffer;

  /* Write header */
  fn__header_t* header = (fn__header_t*)ptr;
  memset(header, 0, sizeof(fn__header_t));

  header->magic[0] = FN_MAGIC_0;
  header->magic[1] = FN_MAGIC_1;
  header->magic[2] = FN_MAGIC_2;
  header->magic[3] = FN_MAGIC_3;
  header->version = FN_FORMAT_VERSION;
  header->data_type = (uint8_t)idx->distance->data_type;
  header->metric_type = (uint8_t)idx->distance->metric;
  header->dimension = idx->distance->dimension;
  header->max_node_count = idx->max_node_count;
  header->cur_num_nodes = idx->cur_num_nodes;
  header->M = idx->M;
  header->data_size_bytes = idx->data_size_bytes;
  header->node_size_bytes = idx->node_size_bytes;
  header->label_size = sizeof(fn_label_t);
  header->flags = 0;

  ptr += sizeof(fn__header_t);

  /* Write node memory */
  memcpy(ptr, idx->node_memory, (size_t)idx->node_size_bytes * idx->max_node_count);

  *buffer_size = required_size;
  return FN_ERR_OK;
}

FN_API fn_error_t fn_index_from_buffer(fn_index_t** out_index, const void* buffer, size_t buffer_size) {
  if (out_index == NULL || buffer == NULL) {
    return FN_ERR_NULL_PTR;
  }

  if (buffer_size < sizeof(fn__header_t)) {
    return FN_ERR_INVALID_FORMAT;
  }

  const char* ptr = (const char*)buffer;

  /* Read header */
  const fn__header_t* header = (const fn__header_t*)ptr;

  /* Validate magic */
  if (header->magic[0] != FN_MAGIC_0 || header->magic[1] != FN_MAGIC_1 || header->magic[2] != FN_MAGIC_2 ||
      header->magic[3] != FN_MAGIC_3) {
    return FN_ERR_INVALID_FORMAT;
  }

  /* Validate version */
  if (header->version != FN_FORMAT_VERSION) {
    return FN_ERR_INVALID_FORMAT;
  }

  /* Validate buffer size */
  size_t expected_size = sizeof(fn__header_t) + (size_t)header->node_size_bytes * header->max_node_count;
  if (buffer_size < expected_size) {
    return FN_ERR_INVALID_FORMAT;
  }

  /* Create index */
  fn_index_config_t config;
  memset(&config, 0, sizeof(config));
  config.dimension = header->dimension;
  config.max_node_count = header->max_node_count;
  config.max_edges_per_node = header->M;
  config.metric = (fn_metric_type_t)header->metric_type;
  config.data_type = (fn_data_type_t)header->data_type;
  config.collect_stats = 0;

  fn_error_t err = fn_index_create(out_index, &config);
  if (err != FN_ERR_OK) {
    return err;
  }

  fn__index_internal_t* idx = (fn__index_internal_t*)(*out_index);

  ptr += sizeof(fn__header_t);

  /* Copy node memory */
  memcpy(idx->node_memory, ptr, (size_t)header->node_size_bytes * header->max_node_count);

  /* Set current node count */
  idx->cur_num_nodes = header->cur_num_nodes;

  return FN_ERR_OK;
}
