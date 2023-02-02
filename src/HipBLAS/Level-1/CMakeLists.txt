add_executable(level1
    amax.cpp
    amin.cpp
    copy.cpp
    main.cpp)

target_include_directories(level1
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../Common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../Common)
target_link_libraries(level1
    PUBLIC
        ${HIPBLAS_LIBS}
    )