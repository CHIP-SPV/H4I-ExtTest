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
#
# We'd like to just use target_compile_features() to specify
# that we need C++ 17 support, and not have to specify
# CMAKE_CXX_STANDARD.  But CMake does not necessarily add
# the -std flag to specify the C++ standard if it has determined
# that the compiler's default is already at or above the 
# standard specified in the target_compile_features().  
# Normally this is not a problem, but in our case where we're
# telling the compiler to compile it as HIP code (i.e., clang++ -x hip),
# CMake's check fails to determine the compiler's default 
# C++ standard level and apparently clang's default C++ level
# when compiling HIP code is *not* the same as if it is compiling C++ code.
# Long story short - we explicitly set CMAKE_CXX_STANDARD to
# force CMake to add the '-std=c++17' flag to the clang++ command line
# regardless of whether we're compiling HIP or straight C++ code.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
add_library(ExtTestConfig INTERFACE)
target_compile_features(ExtTestConfig
    INTERFACE
        cxx_std_17
    )
target_include_directories(ExtTestConfig
    INTERFACE
        ${CMAKE_SOURCE_DIR}/tests/include
        ${CMAKE_BINARY_DIR}/tests/include)

# We will use HIP in some fashion, no matter which platform
# we're targeting or what parts of the software we're building.
find_package(hip REQUIRED)

# The standalone tests should work with either the libraries
# provided with ROCm or our H4I implementations.
option(H4I_USE_ROCM_LIBS "Whether to use ROCm-installed libraries (e.g., ROCm's hipBLAS)" OFF)

# Allow user to exclude half-precision tests.
# Experience shows these seem to be troublesome with CHIP-SPV and the H4I-HipBLAS libraries.
option(TEST_HALF_PRECISION "Whether to include half-precision tests" OFF)

# We (may) use CTest support for driving tests.
include(CTest)

# We use Catch2 to implement (many? most? all?) tests.
find_package(Catch2 3)
if(Catch2_FOUND)
    message(STATUS "Catch2 found at ${Catch2_DIR}")
else()
    # We didn't find Catch2.
    # Rather than fail, download and build it.
    include(FetchContent)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.3.2
    )
    FetchContent_MakeAvailable(Catch2)
endif()
include(Catch)

# The purpose of this repository is to build tests,
# so we don't condition whether we descend into the
# tests directory on the BUILD_TESTING variable.
# NB: we do use BUILD_TESTING later to decide whether to
# register the tests we build with CTest.  We allow this
# type of user control because there are situations where
# the build environment is different than the run environment,
# and so we can't run the Catch2-based executables to list the
# tests they implement in the build environment.
add_subdirectory(tests)

