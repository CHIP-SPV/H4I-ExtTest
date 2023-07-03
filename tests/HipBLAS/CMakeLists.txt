
message(STATUS "HIPBLAS_LIBS=${HIPBLAS_LIBS}")

# Specify how to build BLAS Level 3 tests.
add_executable(blas_level3 level3.cpp)
target_include_directories(blas_level3
    PUBLIC
        ${CMAKE_SOURCE_DIR}/tests/include
        ${CMAKE_BINARY_DIR}/tests/include)
target_link_libraries(blas_level3
    PRIVATE
        ExtTestConfig
    PUBLIC
        ${HIPBLAS_LIBS}
        Catch2::Catch2WithMain)
catch_discover_tests(blas_level3)        

