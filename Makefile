CPP_FILES := $(wildcard flatnav/**/*.h flatnav/**/*.cpp flatnav_python/*.cpp tools/*.cpp developmental-features/**/*.h)

format-cpp:
	clang-format -i $(CPP_FILES)

build-cpp:
	./bin/build.sh -e -t

cmake-format:
	cmake-format -i CMakeLists.txt

run-cpp-unit-tests: build-cpp
	./build/test_distances
	./build/test_serialization

setup-clang-cmake-libomp:
	./bin/install_clang_and_libomp.sh
	./bin/install_cmake.sh