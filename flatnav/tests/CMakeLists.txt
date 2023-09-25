cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
enable_testing()
include(GoogleTest)

# Add test executables here 
set(FLAT_NAV_LIB_TESTS test_serialization)


foreach(TEST IN LISTS FLAT_NAV_LIB_TESTS)
  add_executable(${TEST} ${TEST}.cpp)
  target_link_libraries(${TEST} gtest gtest_main FLAT_NAV_LIB)

  gtest_discover_tests(${TEST} DISCOVERY_TIMEOUT 60)
  install(TARGETS ${TEST} RUNTIME DESTINATION bin)

endforeach(TEST IN LISTS FLAT_NAV_LIB_TESTS)