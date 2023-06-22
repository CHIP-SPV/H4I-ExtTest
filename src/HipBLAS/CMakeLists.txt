# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

# All tests under this director use HipBLAS.
find_package(hipblas REQUIRED)
if(NOT H4I_USE_ROCM_LIBS)
    set(HIPBLAS_LIBS H4I::hipblas)
else()
    # We are using the ROCm hipblas.
    set(HIPBLAS_LIBS roc::hipblas)
endif()

add_subdirectory(Sgemm)

