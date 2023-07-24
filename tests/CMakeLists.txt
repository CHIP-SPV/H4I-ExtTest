# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

configure_file(
    ${CMAKE_SOURCE_DIR}/tests/include/ExtTestConfig.h.in
    ${CMAKE_BINARY_DIR}/tests/include/ExtTestConfig.h)

# Specify how to build our Catch2-based custom main function.
add_library(ExtTestCatch2Main Catch2Main.cpp)
target_link_libraries(ExtTestCatch2Main
    PRIVATE
        ExtTestConfig
    PUBLIC
        Catch2::Catch2)


# Check if we have a hipBLAS implementation.
find_package(hipblas)
if(hipblas_FOUND)
    message(STATUS "Found hipBLAS at ${hipblas_DIR}")

    # We found a hipBLAS implementation.
    if(TARGET H4I::hipblas)
        # We found the H4I hipblas implementation.
        set(HIPBLAS_LIBS H4I::hipblas)
    else()
        # Assume it is the ROCm hipblas.
        set(HIPBLAS_LIBS roc::hipblas ${hip_LIBRARIES})
    endif()

    # Build the hipBLAS tests.
    add_subdirectory(HipBLAS)
endif()

# Check if we have a hipSOLVER implementation.
find_package(hipsolver)
if(hipsolver_FOUND)
    message(STATUS "Found hipSOLVER at ${hipsolver_DIR}")

    # We found a hipSOLVER implementation.
    if(TARGET H4I::hipsolver)
        # We found the H4I hipsolver implementation.
        set(HIPSOLVER_LIBS H4I::hipsolver)
    else()
        # Assume it is the ROCm hipsolver.
        set(HIPSOLVER_LIBS roc::hipsolver ${hip_LIBRARIES})
    endif()

    # Build the hipSOLVER tests.
    add_subdirectory(HipSOLVER)
endif()

