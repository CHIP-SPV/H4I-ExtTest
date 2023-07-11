# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

function(define_hipblas_executable TARGET_NAME TARGET_SOURCES)

    add_executable(${TARGET_NAME} ${TARGET_SOURCES})
    target_link_libraries(${TARGET_NAME}
        PRIVATE
            ExtTestConfig
        PUBLIC
            ${HIPBLAS_LIBS}
            Catch2::Catch2WithMain)

    if(BUILD_TESTING)
        # Register supported tests with CTest.
        catch_discover_tests(${TARGET_NAME})
    endif()
endfunction()

# Specify how to build our test executables.
define_hipblas_executable(hipblas_level3 level3.cpp)
define_hipblas_executable(hipblas_handle_mgmt handle_mgmt.cpp)

# Specify how (and where) to install our targets.
install(TARGETS
            hipblas_level3
            hipblas_handle_mgmt
    RUNTIME)

