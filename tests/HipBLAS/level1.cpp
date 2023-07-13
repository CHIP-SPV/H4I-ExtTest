// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <catch2/generators/catch_generators_all.hpp>

#include "HipBLAS/Level1/AxpyTester.h"
#include "HipBLAS/Level1/DotTester.h"


TEST_CASE("AXPY", "[BLAS][BLAS1]")
{
    auto createCustomStream = GENERATE(false, true);

    HipStream hipStream(createCustomStream);

    H4I::ExtTest::AxpyTester<float>::TestSection(hipStream);
    H4I::ExtTest::AxpyTester<double>::TestSection(hipStream);
}

TEST_CASE("DOT", "[BLAS][BLAS1]")
{
    auto createCustomStream = GENERATE(false, true);

    HipStream hipStream(createCustomStream);

    H4I::ExtTest::DotTester<float>::TestSection(hipStream);
    H4I::ExtTest::DotTester<double>::TestSection(hipStream);
}

