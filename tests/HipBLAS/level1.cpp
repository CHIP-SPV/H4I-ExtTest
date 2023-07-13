// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <catch2/generators/catch_generators_all.hpp>

#include "HipBLAS/Level1/AxpyTester.h"
#include "HipBLAS/Level1/DotTester.h"


TEST_CASE("AXPY", "[BLAS][BLAS1]")
{
    auto createCustomStream = GENERATE(false, true);

    HipStream hipStream(createCustomStream);

    AxpyTester<float>::TestSection(hipStream);
    AxpyTester<double>::TestSection(hipStream);
}

TEST_CASE("DOT", "[BLAS][BLAS1]")
{
    auto createCustomStream = GENERATE(false, true);

    HipStream hipStream(createCustomStream);

    DotTester<float>::TestSection(hipStream);
    DotTester<double>::TestSection(hipStream);
}

