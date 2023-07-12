// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <catch2/generators/catch_generators_all.hpp>

#include "HipBLAS/Level3/GemmTester.h"


TEST_CASE("GEMM", "[BLAS][BLAS3]")
{
    auto createCustomStream = GENERATE(false, true);

    HipStream hipStream(createCustomStream);

    GemmTestSection<float>("sgemm", hipStream);
    GemmTestSection<double>("dgemm", hipStream);
}

