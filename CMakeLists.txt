# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.
cmake_minimum_required(VERSION 3.20)
include(${CMAKE_SOURCE_DIR}/CMake/ExtTestVersion.cmake)
project(ExtTest
    VERSION ${ExtTest_VERSION}
    LANGUAGES CXX)

# Define a target capturing common configuration settings.
# Although we use 'add_library' for this, it is not a library - 
# just a CMake target with a collection of properties set the
# way we want.
# Unfortunately, it doesn't seem to be possible to set all
# of the properties we want on this target and have them
# be inherited by targets that "link" it.
# In particular, we can't set a few C++ standards properties
# and so either have to set them globally or on every target.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(ExtTestConfig INTERFACE)
target_compile_features(ExtTestConfig
    INTERFACE
        cxx_std_17
    )

# We will use HIP in some fashion, no matter which platform
# we're targeting or what parts of the software we're building.
find_package(HIP REQUIRED)
# CMake's support for HIP as a first class language seems
# to assume a ROCm-based implementation.  So we can't use it.
#enable_language(HIP)

# The standalone tests should work with either the libraries
# provided with ROCm or our H4I implementations.
option(H4I_USE_ROCM_LIBS "Whether to use ROCm-installed libraries (e.g., ROCm's hipBLAS)" OFF)

# Allow user to exclude half-precision tests.
# Experience shows these seem to be troublesome with CHIP-SPV and the H4I-HipBLAS libraries.
option(TEST_HALF_PRECISION "Whether to include half-precision tests" OFF)

# Our standalone tests use some Boost libraries.
find_package(Boost REQUIRED
    COMPONENTS program_options)

add_subdirectory(src)

