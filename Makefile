CPP_FILES := $(wildcard flatnav/**/*.h flatnav/**/*.cpp flatnav_python/*.cpp tools/*.cpp developmental-features/**/*.h)

format-cpp:
	clang-format -i $(CPP_FILES)

build-cpp:
	./bin/build.sh -e -t

cmake-format:
	cmake-format -i CMakeLists.txt