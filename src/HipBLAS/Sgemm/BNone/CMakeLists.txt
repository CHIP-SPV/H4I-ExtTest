
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

