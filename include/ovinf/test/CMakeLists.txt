include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../utils
  ${OpenVINO_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
  ${readerwriterqueue_SOURCE_DIR}
)
add_executable(ov_test ov_test.cc)
target_link_libraries(ov_test
  ovinf
)

add_executable(buffer_test buffer_test.cc)
target_link_libraries(buffer_test
  ovinf
)

add_executable(model_verify model_verify.cc)
target_link_libraries(model_verify
  ovinf
)

add_executable(filter_test filter_test.cc)
target_link_libraries(filter_test
  ovinf
)
