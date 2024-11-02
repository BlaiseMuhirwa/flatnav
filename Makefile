CPP_FILES := $(wildcard flatnav/**/*.h flatnav/**/*.cc flatnav/**/*.cpp flatnav_python/*.cpp)

format-cpp:
	clang-format -i $(CPP_FILES)

build-cpp:
	./bin/build.sh -e -t