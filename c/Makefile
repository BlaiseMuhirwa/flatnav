# FlatNav C API Makefile
#
# Targets:
#   make build         - Build the static library (default)
#   make shared        - Build the shared library for Python bindings
#   make python        - Build shared library and copy to Python package
#   make test          - Run unit tests
#   make benchmark     - Run benchmark on SIFT-128 dataset
#   make clean         - Clean build artifacts
#
# Benchmark targets:
#   make sift-bench    - SIFT-128 (1M vectors, 128D, L2)
#   make gist-bench    - GIST-960 (1M vectors, 960D, L2)
#   make glove-bench   - GloVe-100 (1.2M vectors, 100D, angular)
#   make deep-bench    - Deep Image (10M vectors, 96D, angular)

# Configuration
BUILD_DIR := build
CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=Release -DFN_NATIVE_ARCH=ON
NUM_THREADS := 16

# Data directory (relative to this Makefile)
DATA_DIR := ../data

# Python configuration
PYTHON := python3
PYTHON_DIR := python
PYTHON_LIB_DIR := $(PYTHON_DIR)/flatnav_c

# Benchmark configuration
BENCHMARK_SCRIPT := $(PYTHON_DIR)/run_benchmark.py
DEFAULT_M := 32
DEFAULT_EF_CONSTRUCTION := 100 200
DEFAULT_EF_SEARCH := 100 200 300 500 1000
DEFAULT_K := 100
DEFAULT_BUILD_THREADS := $(NUM_THREADS)
DEFAULT_SEARCH_THREADS := 1
DEFAULT_NUM_INIT := 100

#==============================================================================
# Build targets
#==============================================================================

.PHONY: all build shared python test clean help

all: build

format:
	find . -type f \( -name '*.c' -o -name '*.h' \) -exec clang-format -i {} +

# Build static library
build: $(BUILD_DIR)/Makefile
	@echo "Building static library..."
	cmake --build $(BUILD_DIR) -j$(NUM_THREADS)

# Build shared library
shared: $(BUILD_DIR)/Makefile
	@echo "Building shared library..."
	cmake $(BUILD_DIR) -DFN_BUILD_SHARED=ON
	cmake --build $(BUILD_DIR) -j$(NUM_THREADS)

# Build Python bindings (shared library + copy to package)
python: shared
	@echo "Copying shared library to Python package..."
	cp $(BUILD_DIR)/libflatnav_c.so $(PYTHON_LIB_DIR)/
	@echo "Python bindings ready at $(PYTHON_LIB_DIR)"
	@echo ""
	@echo "To use: cd $(PYTHON_DIR) && python3 -c 'import flatnav_c; print(flatnav_c.build_info())'"

# Create build directory and run CMake
$(BUILD_DIR)/Makefile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. $(CMAKE_FLAGS)

# Run tests
test: build
	@echo "Running tests..."
	cd $(BUILD_DIR) && ctest --output-on-failure

# Run C benchmark test (requires SIFT data)
test-benchmark: build
	@echo "Running C benchmark test..."
	cd $(BUILD_DIR) && ./test_benchmark

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(PYTHON_LIB_DIR)/libflatnav_c.so

#==============================================================================
# Python Benchmark targets
#==============================================================================

.PHONY: benchmark sift-bench gist-bench glove-bench deep-bench

# Quick benchmark on SIFT-128 with reduced dataset
benchmark: python
	@echo "Running quick benchmark on SIFT-128 (100K vectors)..."
	$(PYTHON) $(BENCHMARK_SCRIPT) \
		--dataset-name sift-128-quick \
		--dataset $(DATA_DIR)/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries $(DATA_DIR)/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth $(DATA_DIR)/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--metric l2 \
		--num-node-links $(DEFAULT_M) \
		--ef-construction 100 200 \
		--ef-search 100 200 300 \
		--num-build-threads $(DEFAULT_BUILD_THREADS) \
		--num-search-threads $(DEFAULT_SEARCH_THREADS) \
		--max-vectors 1000000

#------------------------------------------------------------------------------
# SIFT-128 Benchmark (1M vectors, 128D, L2)
#------------------------------------------------------------------------------

sift-bench: python
	@echo "Running SIFT-128 benchmark..."
	$(PYTHON) $(BENCHMARK_SCRIPT) \
		--dataset-name sift-128 \
		--dataset $(DATA_DIR)/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries $(DATA_DIR)/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth $(DATA_DIR)/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--metric l2 \
		--num-node-links $(DEFAULT_M) \
		--ef-construction $(DEFAULT_EF_CONSTRUCTION) \
		--ef-search $(DEFAULT_EF_SEARCH) \
		--k $(DEFAULT_K) \
		--num-build-threads $(DEFAULT_BUILD_THREADS) \
		--num-search-threads $(DEFAULT_SEARCH_THREADS) \
		--num-init $(DEFAULT_NUM_INIT) \
		--output results/sift-128-results.json

sift-bench-full: python
	@mkdir -p results
	@echo "Running SIFT-128 full benchmark..."
	$(PYTHON) $(BENCHMARK_SCRIPT) \
		--dataset-name sift-128-full \
		--dataset $(DATA_DIR)/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries $(DATA_DIR)/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth $(DATA_DIR)/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--metric l2 \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--k 10 \
		--num-build-threads $(DEFAULT_BUILD_THREADS) \
		--num-search-threads 1 \
		--output results/sift-128-full-results.json

#------------------------------------------------------------------------------
# GIST-960 Benchmark (1M vectors, 960D, L2)
#------------------------------------------------------------------------------

gist-bench: python
	@mkdir -p results
	@echo "Running GIST-960 benchmark..."
	$(PYTHON) $(BENCHMARK_SCRIPT) \
		--dataset-name gist-960 \
		--dataset $(DATA_DIR)/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries $(DATA_DIR)/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth $(DATA_DIR)/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--metric l2 \
		--num-node-links $(DEFAULT_M) \
		--ef-construction $(DEFAULT_EF_CONSTRUCTION) \
		--ef-search $(DEFAULT_EF_SEARCH) \
		--k $(DEFAULT_K) \
		--num-build-threads $(DEFAULT_BUILD_THREADS) \
		--num-search-threads $(DEFAULT_SEARCH_THREADS) \
		--output results/gist-960-results.json

#------------------------------------------------------------------------------
# GloVe-100 Benchmark (1.2M vectors, 100D, angular/IP)
#------------------------------------------------------------------------------

glove-bench: python
	@mkdir -p results
	@echo "Running GloVe-100 benchmark..."
	$(PYTHON) $(BENCHMARK_SCRIPT) \
		--dataset-name glove-100-angular \
		--dataset $(DATA_DIR)/glove-100-angular/glove-100-angular.train.npy \
		--queries $(DATA_DIR)/glove-100-angular/glove-100-angular.test.npy \
		--gtruth $(DATA_DIR)/glove-100-angular/glove-100-angular.gtruth.npy \
		--metric angular \
		--num-node-links $(DEFAULT_M) \
		--ef-construction $(DEFAULT_EF_CONSTRUCTION) \
		--ef-search $(DEFAULT_EF_SEARCH) \
		--k $(DEFAULT_K) \
		--num-build-threads $(DEFAULT_BUILD_THREADS) \
		--num-search-threads $(DEFAULT_SEARCH_THREADS) \
		--output results/glove-100-results.json

#------------------------------------------------------------------------------
# Deep Image-96 Benchmark (10M vectors, 96D, angular)
#------------------------------------------------------------------------------

deep-bench: python
	@mkdir -p results
	@echo "Running Deep Image-96 benchmark..."
	$(PYTHON) $(BENCHMARK_SCRIPT) \
		--dataset-name deep-image-96 \
		--dataset $(DATA_DIR)/deep-image-96-angular/deep-image-96-angular.train.npy \
		--queries $(DATA_DIR)/deep-image-96-angular/deep-image-96-angular.test.npy \
		--gtruth $(DATA_DIR)/deep-image-96-angular/deep-image-96-angular.gtruth.npy \
		--metric angular \
		--num-node-links $(DEFAULT_M) \
		--ef-construction $(DEFAULT_EF_CONSTRUCTION) \
		--ef-search $(DEFAULT_EF_SEARCH) \
		--k $(DEFAULT_K) \
		--num-build-threads $(DEFAULT_BUILD_THREADS) \
		--num-search-threads $(DEFAULT_SEARCH_THREADS) \
		--output results/deep-image-96-results.json

#------------------------------------------------------------------------------
# All benchmarks
#------------------------------------------------------------------------------

.PHONY: all-benchmarks

all-benchmarks: sift-bench gist-bench glove-bench deep-bench
	@echo ""
	@echo "All benchmarks complete. Results saved to results/"

#==============================================================================
# Help
#==============================================================================

help:
	@echo "FlatNav C API Makefile"
	@echo ""
	@echo "Build targets:"
	@echo "  make build         - Build the static library (default)"
	@echo "  make shared        - Build the shared library"
	@echo "  make python        - Build shared library for Python bindings"
	@echo "  make test          - Run unit tests"
	@echo "  make test-benchmark - Run C benchmark test (requires SIFT data)"
	@echo "  make clean         - Clean build artifacts"
	@echo ""
	@echo "Benchmark targets:"
	@echo "  make benchmark     - Quick benchmark on SIFT-128 (100K vectors)"
	@echo "  make sift-bench    - SIFT-128 full benchmark (1M vectors)"
	@echo "  make gist-bench    - GIST-960 benchmark (1M vectors)"
	@echo "  make glove-bench   - GloVe-100 benchmark (1.2M vectors)"
	@echo "  make deep-bench    - Deep Image-96 benchmark (10M vectors)"
	@echo "  make all-benchmarks - Run all benchmarks"
	@echo ""
	@echo "Configuration variables (override with make VAR=value):"
	@echo "  NUM_THREADS=$(NUM_THREADS)"
	@echo "  DEFAULT_M=$(DEFAULT_M)"
	@echo "  DEFAULT_BUILD_THREADS=$(DEFAULT_BUILD_THREADS)"
	@echo "  DEFAULT_SEARCH_THREADS=$(DEFAULT_SEARCH_THREADS)"
	@echo "  DATA_DIR=$(DATA_DIR)"
