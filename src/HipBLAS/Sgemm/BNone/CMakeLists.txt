# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

add_executable(sgemm_hb_none
    main.cpp)

target_include_directories(sgemm_hb_none
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../Common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../Common)
target_link_libraries(sgemm_hb_none
    PRIVATE
        ExtTestConfig
    PUBLIC
        Boost::program_options
        ${HIPBLAS_LIBS}
    )

