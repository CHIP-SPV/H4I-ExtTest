// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <catch2/generators/catch_generators_all.hpp>

#if READY
#include "HipBLAS/Level1/AxpyTester.h"
#endif // READY
#include "HipBLAS/Level1/DotTester.h"


#if READY
TEST_CASE("AXPY", "[BLAS][BLAS1]")
{
    auto createCustomStream = GENERATE(false, true);

    HipStream hipStream(createCustomStream);

    AxpyTester<float>::TestSection(hipStream);
    AxpyTester<double>::TestSection(hipStream);
}
#endif // READY

TEST_CASE("DOT", "[BLAS][BLAS1]")
{
    auto createCustomStream = GENERATE(false, true);

    HipStream hipStream(createCustomStream);

    DotTester<float>::TestSection(hipStream);
//     DotTester<double>::TestSection(hipStream);
}

