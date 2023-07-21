# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

function(define_hipsolver_executable TARGET_NAME TARGET_SOURCES)

    add_executable(${TARGET_NAME} ${TARGET_SOURCES})
    target_link_libraries(${TARGET_NAME}
        PRIVATE
            ExtTestConfig
        PUBLIC
            ${HIPSOLVER_LIBS}
            Catch2::Catch2WithMain)

    if(BUILD_TESTING)
        # Register supported tests with CTest.
        catch_discover_tests(${TARGET_NAME})
    endif()
endfunction()

# Specify how to build our test executables.
define_hipsolver_executable(hipsolver_handle_mgmt handle_mgmt.cpp)
define_hipsolver_executable(hipsolver_linear_solve linear_solve.cpp)

# Specify how (and where) to install our targets.
install(TARGETS
            hipsolver_handle_mgmt
            hipsolver_linear_solve
    RUNTIME)

