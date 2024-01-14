cmake_minimum_required(VERSION 3.14 FATAL_ERROR)


set(EXAMPLES construct_npy query_npy cereal_tests)
foreach(EXAMPLE IN LISTS EXAMPLES)
  add_executable(${EXAMPLE} ${EXAMPLE}.cpp ${HEADERS})
  target_link_libraries(${EXAMPLE} FLAT_NAV_LIB ${CNPY_LIB} ${ZLIB_LIB_RELEASE})

  # This ensures that the executables are placed in the build directory
  set_target_properties(${EXAMPLE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                              "${CMAKE_BINARY_DIR}")
  install(TARGETS ${EXAMPLE} RUNTIME DESTINATION bin)
endforeach(EXAMPLE IN LISTS EXAMPLES)
