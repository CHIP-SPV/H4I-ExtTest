// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "Catch2Session.h"
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

    // Disallow types for which we don't specialize.
    template<typename T>
    inline static const std::string opname;

    template<typename T>
    static auto copy(void) = delete;

    // Specializations for float.
    template<>
    inline static const std::string opname<float> = "scopy";

    template<>
    static auto copy<float>(void)   { return hipblasScopy; }

    // Specializations for double.
    template<>
    inline static const std::string opname<double> = "dcopy";

    template<>
    static auto copy<double>(void)  { return hipblasDcopy; }    


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
            tester.Check();
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
        auto opfunc = copy<ScalarType>();
        HBCHECK(opfunc(this->libContext.GetHandle(),
                            n,
                            x.GetDeviceData(),
                            x.GetIncrement(),
                            y.GetDeviceData(),
                            y.GetIncrement()));
        y.CopyDeviceToHostAsync(this->hipStream);

        this->hipStream.Synchronize();
    }

    void Check(void) const override
    {
        for(auto i = 0; i < n; ++i)
        {
            REQUIRE_THAT(y.El(i), Catch::Matchers::WithinRel(x.El(i), Catch2Session::theSession->GetRelErrThreshold<ScalarType>()));
        }
    }

    // Declare a Catch2 section for a test.
    static void TestSection(HipStream& hipStream)
    {
        // This generic version should never be called.
        // Specializations are provided later.
        using TesterType = CopyTester<ScalarType>;

        SECTION(opname<ScalarType>)
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
            tester.Check();
        }
    }
};

} // namespace

