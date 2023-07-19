// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <catch2/generators/catch_generators_all.hpp>

#include "HipBLAS/Level1/AxpyTester.h"
#include "HipBLAS/Level1/DotTester.h"
#include "HipBLAS/Level1/CopyTester.h"
#include "HipBLAS/Level1/IXamaxTester.h"
#include "HipBLAS/Level1/IXaminTester.h"


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

TEST_CASE("COPY", "[BLAS][BLAS1]")
{
    auto createCustomStream = GENERATE(false, true);

    HipStream hipStream(createCustomStream);

    H4I::ExtTest::CopyTester<float>::TestSection(hipStream);
    H4I::ExtTest::CopyTester<double>::TestSection(hipStream);
}

TEST_CASE("IXAMAX", "[BLAS][BLAS1]")
{
    auto createCustomStream = GENERATE(false, true);

    HipStream hipStream(createCustomStream);

    H4I::ExtTest::IXamaxTester<float>::TestSection(hipStream);
    H4I::ExtTest::IXamaxTester<double>::TestSection(hipStream);
}

TEST_CASE("IXAMIN", "[BLAS][BLAS1]")
{
    auto createCustomStream = GENERATE(false, true);

    HipStream hipStream(createCustomStream);

    H4I::ExtTest::IXaminTester<float>::TestSection(hipStream);
    H4I::ExtTest::IXaminTester<double>::TestSection(hipStream);
}

