// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <catch2/catch_test_macros.hpp>

#include "hipsolver.h"


TEST_CASE("HipSOLVER handle management", "[SOLVER][util]")
{
    SECTION("Default HIP stream")
    {
        // Create a HipSOLVER context with default stream.
        hipsolverHandle_t hbh;
        auto hbstatus = hipsolverCreate(&hbh);
        REQUIRE(hbstatus == HIPSOLVER_STATUS_SUCCESS);

        // Verify that context uses default stream.
        hipStream_t hsh;
        hbstatus = hipsolverGetStream(hbh, &hsh);
        REQUIRE(hbstatus == HIPSOLVER_STATUS_SUCCESS);
        REQUIRE(hsh == nullptr);

        // Clean up.
        hbstatus = hipsolverDestroy(hbh);
        REQUIRE(hbstatus == HIPSOLVER_STATUS_SUCCESS);
    }

    SECTION("Custom HIP stream")
    {
        // Create a HipSOLVER context with default stream.
        hipsolverHandle_t hbh;
        auto hbstatus = hipsolverCreate(&hbh);
        REQUIRE(hbstatus == HIPSOLVER_STATUS_SUCCESS);

        // Create a "custom" HIP stream.
        hipStream_t hsh;
        auto hsstatus = hipStreamCreate(&hsh);
        REQUIRE(hsstatus == HIP_SUCCESS);

        // Tell hipSOLVER to use the custom stream.
        hbstatus = hipsolverSetStream(hbh, hsh);
        REQUIRE(hbstatus == HIPSOLVER_STATUS_SUCCESS);

        // Verify it is using the custom stream.
        hipStream_t testh;
        hbstatus = hipsolverGetStream(hbh, &testh);
        REQUIRE(hbstatus == HIPSOLVER_STATUS_SUCCESS);
        REQUIRE(testh == hsh);

        // Clean up.
        hbstatus = hipsolverDestroy(hbh);
        REQUIRE(hbstatus == HIPSOLVER_STATUS_SUCCESS);

        hsstatus = hipStreamDestroy(hsh);
        REQUIRE(hsstatus == HIP_SUCCESS);
    }
}

