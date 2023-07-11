// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <catch2/catch_test_macros.hpp>

#include "hipblas.h"


TEST_CASE("HipBLAS handle management", "[BLAS][util]")
{
    SECTION("Default HIP stream")
    {
        // Create a HipBLAS context with default stream.
        hipblasHandle_t hbh;
        auto hbstatus = hipblasCreate(&hbh);
        REQUIRE(hbstatus == HIPBLAS_STATUS_SUCCESS);

        // Verify that context uses default stream.
        hipStream_t hsh;
        hbstatus = hipblasGetStream(hbh, &hsh);
        REQUIRE(hbstatus == HIPBLAS_STATUS_SUCCESS);
        REQUIRE(hsh == nullptr);

        // Clean up.
        hbstatus = hipblasDestroy(hbh);
        REQUIRE(hbstatus == HIPBLAS_STATUS_SUCCESS);
    }

    SECTION("Custom HIP stream")
    {
        // Create a HipBLAS context with default stream.
        hipblasHandle_t hbh;
        auto hbstatus = hipblasCreate(&hbh);
        REQUIRE(hbstatus == HIPBLAS_STATUS_SUCCESS);

        // Create a "custom" HIP stream.
        hipStream_t hsh;
        auto hsstatus = hipStreamCreate(&hsh);
        REQUIRE(hsstatus == HIP_SUCCESS);

        // Tell hipBLAS to use the custom stream.
        hbstatus = hipblasSetStream(hbh, hsh);
        REQUIRE(hbstatus == HIPBLAS_STATUS_SUCCESS);

        // Verify it is using the custom stream.
        hipStream_t testh;
        hbstatus = hipblasGetStream(hbh, &testh);
        REQUIRE(hbstatus == HIPBLAS_STATUS_SUCCESS);
        REQUIRE(testh == hsh);

        // Clean up.
        hbstatus = hipblasDestroy(hbh);
        REQUIRE(hbstatus == HIPBLAS_STATUS_SUCCESS);

        hsstatus = hipStreamDestroy(hsh);
        REQUIRE(hsstatus == HIP_SUCCESS);
    }
}

