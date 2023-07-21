// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "HipBLAS/HipblasTester.h"
#include "Vector.h"
#include "Scalar.h"


namespace H4I::ExtTest
{

// A class to test the BLAS Level 1 copy operation.
// The result should be:
// y[i] = x[i], where x has stride incx and y has stride incy.
//
template<typename ScalarType>
class CopyTester : public HipblasTester<ScalarType>
{
private:
    int n;
    Vector<ScalarType> x;
    Vector<ScalarType> y;

    // A scaling factor so initial values are not integers.
    // *NOT* part of the BLAS definition of the operation.
    ScalarType scalingFactor;

    static hipblasStatus_t CallCopy(hipblasHandle_t handle,
                                int n,
                                const ScalarType* x,
                                int incx,
                                ScalarType* y,
                                int incy)
    {
        // This generic version should never be called.
        // Specializations will be provided later.
        assert(false);
    }

    static void TestSectionAux(std::string sectionName, HipStream& hipStream)
    {
        using TesterType = CopyTester<ScalarType>;

        SECTION(sectionName)
        {
            // Specify the problem.
            int n = GENERATE(take(1, random(50, 150)));
            int incx = GENERATE(1, 4);
            int incy = GENERATE(1, 7);
            ScalarType factor = GENERATE(take(1, random(0.0, 1.5)));

            // Build a test driver.
            TesterType tester(n, incx, incy, factor, hipStream);
            REQUIRE_NOTHROW(tester.Init());

            // Do the operation.
            REQUIRE_NOTHROW(tester.DoOperation());

            // Verify the result.
            ScalarType relErrTolerance = 0.0001;
            tester.Check(relErrTolerance);
        }
    }

public:
    CopyTester(int _n,
                int _incx,
                int _incy,
                ScalarType _factor,
                HipStream& _hipStream)
      : HipblasTester<ScalarType>(_hipStream),
        n(_n),
        x(_n, _incx),
        y(_n, _incy),
        scalingFactor(_factor)
    { }

    // Create the input with known values.  See the comment
    // above for the description of the input values and
    // expected result.
    void Init(void) override
    {
        for(auto i = 0; i < x.GetNumItems(); ++i)
        {
            x.El(i) = scalingFactor*i;
        }
        x.CopyHostToDeviceAsync(this->hipStream);

        this->hipStream.Synchronize();
    }


    void DoOperation(void) override
    {
        HBCHECK(CallCopy(this->libContext.GetHandle(),
                            n,
                            x.GetDeviceData(),
                            x.GetIncrement(),
                            y.GetDeviceData(),
                            y.GetIncrement()));
        y.CopyDeviceToHostAsync(this->hipStream);

        this->hipStream.Synchronize();
    }

    void Check(ScalarType relErrTolerance) const override
    {
        for(auto i = 0; i < n; ++i)
        {
            REQUIRE_THAT(y.El(i), Catch::Matchers::WithinRel(x.El(i), relErrTolerance));
        }
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
CopyTester<float>::CallCopy(hipblasHandle_t handle,
                            int n,
                            const float* x,
                            int incx,
                            float* y,
                            int incy)
{
    return hipblasScopy(handle,
            n,
            x,
            incx,
            y,
            incy);
}


// Double-precision operation.
template<>
hipblasStatus_t
CopyTester<double>::CallCopy(hipblasHandle_t handle,
                            int n,
                            const double* x,
                            int incx,
                            double* y,
                            int incy)
{
    return hipblasDcopy(handle,
            n,
            x,
            incx,
            y,
            incy);
}


template<>
void
CopyTester<float>::TestSection(HipStream& hipStream)
{
    TestSectionAux("scopy", hipStream);
}

template<>
void
CopyTester<double>::TestSection(HipStream& hipStream)
{
    TestSectionAux("dcopy", hipStream);
}

} // namespace

