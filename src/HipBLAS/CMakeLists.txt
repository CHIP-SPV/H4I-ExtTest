# Copyright 2021-2022 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

# All tests under this director use HipBLAS.
if(NOT H4I_USE_ROCM_LIBS)
    find_package(HipBLAS)
    set(HIPBLAS_LIBS H4I::HipBLAS)
else()
    # We are using the ROCm hipblas.
    find_package(hipblas REQUIRED)
    set(HIPBLAS_LIBS roc::hipblas)
endif()

add_subdirectory(Sgemm)

