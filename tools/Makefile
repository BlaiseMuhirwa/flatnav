# tools/Makefile — convenience targets for building and benchmarking The C++ flatnav API
#
# Usage:
#   cd tools
#   make build                           # build with examples/tools enabled
#   make sift-128-euclidean              # construct + query (no quantization)
#   make sift-128-euclidean QUANT=2      # construct + query with scalar quantization
#   make bench-all QUANT=2               # run a few key datasets with SQ
#
# All paths are relative to the repo root (one level up from tools/).

ROOT     := $(shell cd .. && pwd)
BUILD    := $(ROOT)/build
DATA     := $(ROOT)/data

# Configurable parameters
QUANT            ?= 0
M                ?= 32
EF_CONSTRUCTION  ?= 100
EF_SEARCH        ?= 100,200,300
K                ?= 100
THREADS          ?= 16
REORDER          ?= 0

# Binaries (built by cmake)
CONSTRUCT := $(BUILD)/construct_npy
QUERY     := $(BUILD)/query_npy

# Determine metric from dataset name: "euclidean" -> 0 (L2), otherwise -> 1 (IP)
metric = $(if $(findstring euclidean,$(1)),0,1)

# Index filename includes quantization mode to avoid overwriting
index_suffix = $(if $(filter 2,$(QUANT)),sq_index,$(if $(filter 1,$(QUANT)),pq_index,index))

# Build
.PHONY: build
build:
	cd $(ROOT) && ./bin/build.sh -e


DATASETS := sift-128-euclidean glove-100-angular gist-960-euclidean deep-image-96-angular

# construct-DATASET
define construct_rule
.PHONY: construct-$(1)
construct-$(1):
	@echo "=== Constructing index for $(1) (QUANT=$(QUANT)) ==="
	$(CONSTRUCT) \
		$(QUANT) \
		$(call metric,$(1)) \
		$(DATA)/$(1)/$(1).train.npy \
		$(M) \
		$(EF_CONSTRUCTION) \
		$(THREADS) \
		$(DATA)/$(1)/$(1).$(call index_suffix)
endef

# query-DATASET
define query_rule
.PHONY: query-$(1)
query-$(1):
	@echo "=== Querying index for $(1) (QUANT=$(QUANT)) ==="
	$(QUERY) \
		$(call metric,$(1)) \
		$(DATA)/$(1)/$(1).$(call index_suffix) \
		$(DATA)/$(1)/$(1).test.npy \
		$(DATA)/$(1)/$(1).gtruth.npy \
		$(EF_SEARCH) \
		$(K) \
		$(REORDER) \
		$(QUANT)
endef

# DATASET (construct + query)
define dataset_rule
.PHONY: $(1)
$(1): construct-$(1) query-$(1)
endef

$(foreach ds,$(DATASETS),$(eval $(call construct_rule,$(ds))))
$(foreach ds,$(DATASETS),$(eval $(call query_rule,$(ds))))
$(foreach ds,$(DATASETS),$(eval $(call dataset_rule,$(ds))))


.PHONY: bench-all
bench-all: sift-128-euclidean glove-100-angular gist-960-euclidean deep-image-96-angular
