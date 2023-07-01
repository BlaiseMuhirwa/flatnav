enable_testing()
include(GoogleTest)

# Add test executables here 
set(PQ_TESTS test_kmeans)

foreach(PQ_TEST IN LISTS PQ_TESTS)
  add_executable(${PQ_TEST} ${PQ_TEST}.cpp)
  target_link_libraries(${PQ_TEST} PRIVATE FLAT_NAV_LIB gtest_main)

  gtest_discover_tests(${PQ_TEST} DISCOVERY_TIMEOUT 60)
  install(TARGETS ${PQ_TEST} RUNTIME DESTINATION BIN)

endforeach(PQ_TEST IN LISTS PQ_TESTS)