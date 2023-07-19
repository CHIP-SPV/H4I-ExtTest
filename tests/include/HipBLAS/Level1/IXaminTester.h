// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "HipBLAS/HipblasTester.h"
#include "Vector.h"


namespace H4I::ExtTest
{

// A class to test the BLAS Level 1 IXamin operation.
// The input values are:
// x[i] = (random value of type ScalarType) with stride incx
//
// The result should be the index of the largest value in x.
//
template<typename ScalarType>
class IXaminTester : public HipblasTester<ScalarType>
{
private:
    int n;
    Vector<ScalarType> x;
    int result;

    static hipblasStatus_t CallIXamin(hipblasHandle_t handle,
                                int n,
                                const ScalarType* x,
                                int incx,
                                int* result)
    {
        // This generic version should never be called.
        // Specializations will be provided later.
        assert(false);
    }

    static void TestSectionAux(std::string sectionName, HipStream& hipStream)
    {
        using TesterType = IXaminTester<ScalarType>;

        SECTION(sectionName)
        {
            // Specify the problem.
            int n = GENERATE(take(1, random(50, 150)));
            int incx = GENERATE(1, 4);

            // Build a test driver.
            TesterType tester(n, incx, hipStream);
            REQUIRE_NOTHROW(tester.Init());

            // Do the operation.
            REQUIRE_NOTHROW(tester.DoOperation());

            // Verify the result.
            // The result should be exact.
            ScalarType relErrTolerance = 0.0;
            tester.Check(relErrTolerance);
        }
    }

public:
    IXaminTester(int _n,
                int _incx,
                HipStream& _hipStream)
      : HipblasTester<ScalarType>(_hipStream),
        n(_n),
        x(_n, _incx),
        result(-1)
    { }

    // Create the input with known values.  See the comment
    // above for the description of the input values and
    // expected result.
    void Init(void) override
    {
        // Generate some random input using the Catch2 generator.
        auto xvals = GENERATE_COPY(chunk(n, take(n, random<ScalarType>(-100.0, 100.0))));

        // Copy the random input into the x Vector.
        // NB: we can't just memcpy or assign entire vectors due to the potential
        // for an X stride that isn't 1.
        for(auto i = 0; i < n; ++i)
        {
            x.El(i) = xvals[i];
        }
        x.CopyHostToDeviceAsync(this->hipStream);

        this->hipStream.Synchronize();
    }


    void DoOperation(void) override
    {
        HBCHECK(CallIXamin(this->blasContext.GetHandle(),
                            n,
                            x.GetDeviceData(),
                            x.GetIncrement(),
                            &result));

        this->hipStream.Synchronize();
    }

    void Check([[maybe_unused]] ScalarType relErrTolerance) const override
    {
        // Find the index of the min absolute value in x on the CPU.
        int minValueIdx = 0;
        ScalarType minValue = std::abs(x.El(minValueIdx));
        for(auto i = 0; i < n; ++i)
        {
            auto testVal = std::abs(x.El(i));
            if(testVal < minValue)
            {
                minValue = testVal;
                minValueIdx = i;
            }
        }

        // Verify that hipBLAS found the same index we just did.
        // NB: hipBLAS is using 1-based index, and we found the 
        // 0-based index on the CPU, so we need to take that 
        // into account when comparing the two.
        REQUIRE(result == (minValueIdx + 1));
    }

    // Declare a Catch2 section for a test.
    static void TestSection(HipStream& hipStream)
    {
        // This generic version should never be called.
        // Specializations are provided later.
        assert(false);
    }
};


// Single-precision operation.
template<>
hipblasStatus_t
IXaminTester<float>::CallIXamin(hipblasHandle_t handle,
                            int n,
                            const float* x,
                            int incx,
                            int* result)
{
    return hipblasIsamin(handle,
            n,
            x,
            incx,
            result);
}


// Double-precision operation.
template<>
hipblasStatus_t
IXaminTester<double>::CallIXamin(hipblasHandle_t handle,
                            int n,
                            const double* x,
                            int incx,
                            int* result)
{
    return hipblasIdamin(handle,
            n,
            x,
            incx,
            result);
}


template<>
void
IXaminTester<float>::TestSection(HipStream& hipStream)
{
    TestSectionAux("isamin", hipStream);
}

template<>
void
IXaminTester<double>::TestSection(HipStream& hipStream)
{
    TestSectionAux("idamin", hipStream);
}

} // namespace
