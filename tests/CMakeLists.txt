
configure_file(
    ${CMAKE_SOURCE_DIR}/tests/include/ExtTestConfig.h.in
    ${CMAKE_BINARY_DIR}/tests/include/ExtTestConfig.h)

# All tests under this directory use HipBLAS.
find_package(hipblas REQUIRED)
if(NOT H4I_USE_ROCM_LIBS)
    set(HIPBLAS_LIBS H4I::hipblas)
else()
    # We are using the ROCm hipblas.
    set(HIPBLAS_LIBS roc::hipblas)
endif()

# add_subdirectory(src)
add_subdirectory(HipBLAS)

