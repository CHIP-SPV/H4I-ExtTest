# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/ExtTestConfig.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Common/ExtTestConfig.h)

add_subdirectory(HipBLAS)

