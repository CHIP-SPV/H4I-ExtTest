// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <catch2/generators/catch_generators_all.hpp>

#include "HipSOLVER/LinearSolve/GesvTester.h"


TEST_CASE("GESV", "[LAPACK][LinearSolve]")
{
    auto createCustomStream = GENERATE(false, true);

    HipStream hipStream(createCustomStream);

    H4I::ExtTest::GesvTester<float>::TestSection(hipStream);
    H4I::ExtTest::GesvTester<double>::TestSection(hipStream);
}

