// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <catch2/generators/catch_generators_all.hpp>

#include "HipBLAS/Level2/GemvTester.h"


TEST_CASE("GEMV", "[BLAS][BLAS2]")
{
    auto createCustomStream = GENERATE(false, true);

    HipStream hipStream(createCustomStream);

    H4I::ExtTest::GemvTester<float>::TestSection(hipStream);
    H4I::ExtTest::GemvTester<double>::TestSection(hipStream);
}

