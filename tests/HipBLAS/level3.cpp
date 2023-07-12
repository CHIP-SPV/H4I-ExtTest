// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <vector>
#include <tuple>

#include "HipBLAS/GemmTester.h"


// Declare a Catch2 section for a GEMM test.
template<typename ScalarType>
void
GemmTestSection(std::string sectionName, HipStream& hipStream)
{
    using TesterType = GemmTester<ScalarType>;

    SECTION(sectionName)
    {
        // Specify the problem.
        // A: m x k
        // B: k x n
        // C: m x n
        int m = GENERATE(take(1, random(50, 150)));
        int n = GENERATE(take(1, random(50, 150)));
        int k = GENERATE(take(1, random(50, 150)));
        ScalarType alpha = 0.5; GENERATE(take(1, random(-1.0, 1.0)));
        ScalarType beta = 0.75; GENERATE(take(1, random(-2.5, 2.5)));
        auto transB = GENERATE(false, true);

        // Build a test driver.
        TesterType tester(m, n, k, alpha, beta, transB, hipStream);
        REQUIRE_NOTHROW(tester.Init());

        // Do the operation.
        REQUIRE_NOTHROW(tester.DoOperation());
        hipStream.Synchronize();

        // Verify the result.
        ScalarType relErrTolerance = 0.0001;
        REQUIRE(tester.Check(relErrTolerance));
    }
}

TEST_CASE("GEMM", "[BLAS][BLAS3]")
{
    auto createCustomStream = GENERATE(false, true);

    HipStream hipStream(createCustomStream);

    GemmTestSection<float>("sgemm", hipStream);
    GemmTestSection<double>("dgemm", hipStream);
}

