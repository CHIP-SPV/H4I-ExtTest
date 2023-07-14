# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

configure_file(
    ${CMAKE_SOURCE_DIR}/tests/include/ExtTestConfig.h.in
    ${CMAKE_BINARY_DIR}/tests/include/ExtTestConfig.h)

# All tests under this directory use HipBLAS.
find_package(hipblas REQUIRED)
if(NOT H4I_USE_ROCM_LIBS)
    set(HIPBLAS_LIBS H4I::hipblas)
else()
    # We are using the ROCm hipblas.
    set(HIPBLAS_LIBS roc::hipblas ${hip_LIBRARIES})
endif()

# add_subdirectory(src)
add_subdirectory(HipBLAS)

