CPP_FILES := $(wildcard flatnav/**/*.h flatnav/**/*.cpp python-bindings/*.cpp tools/*.cpp developmental-features/**/*.h)
CIBUILDWHEEL_VERSION := 2.22.0


format-cpp:
	clang-format -i $(CPP_FILES)

build-cpp:
	./bin/build.sh -e -t

cmake-format:
	cmake-format -i CMakeLists.txt

run-cpp-unit-tests: build-cpp
	./build/test_distances
	./build/test_serialization

install-cibuildwheel:
	pip install "cibuildwheel==${CIBUILDWHEEL_VERSION}"

run-python-unit-tests:
	cd python-bindings && pytest -vs unit_tests

setup-clang-cmake-libomp:
	./bin/install_clang_and_libomp.sh
	./bin/install_cmake.sh