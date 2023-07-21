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

// A class to test the BLAS Level 1 AXPY
// The input vectors are:
// x[i] = i with stride incx
// y[i] = 2*i with stride incy
//
// By the definition of the tested operation,
// the result left in vector Y should be
//   y[i] = a*i + 2*i = (a+2)*i
//
template<typename ScalarType>
class AxpyTester : public HipblasTester<ScalarType>
{
private:
    int n;
    Scalar<ScalarType> alpha;
    Vector<ScalarType> x;
    Vector<ScalarType> y;

    // Disallow types for which we don't specialize.
    template<typename T>
    inline static const std::string opname;

    template<typename T>
    static auto axpy(void) = delete;

    // Specializations for float.
    template<>
    inline static const std::string opname<float> = "saxpy";

    template<>
    static auto axpy<float>(void) { return hipblasSaxpy; }

    // Specializations for double.
    template<>
    inline static const std::string opname<double> = "daxpy";

    template<>
    static auto axpy<double>(void) { return hipblasDaxpy; }

public:
    AxpyTester(int _n,
                ScalarType _alpha,
                int _incx,
                int _incy,
                HipStream& _hipStream)
      : HipblasTester<ScalarType>(_hipStream),
        n(_n),
        alpha(_alpha),
        x(_n, _incx),
        y(_n, _incy)
    { }

    // Create the input with known values.  See the comment
    // above for the description of the input values and
    // expected result.
    void Init(void) override
    {
        for(auto i = 0; i < x.GetNumItems(); ++i)
        {
            x.El(i) = i;
        }
        x.CopyHostToDeviceAsync(this->hipStream);

        for(auto i = 0; i < y.GetNumItems(); ++i)
        {
            y.El(i) = 2*i;
        }
        y.CopyHostToDeviceAsync(this->hipStream);

        // Alpha should already have its value in host memory.
        alpha.CopyHostToDeviceAsync(this->hipStream);

        this->hipStream.Synchronize();
    }


    void DoOperation(void) override
    {
#if READY
        // We need the hipBLAS to assume our scalars
        // are in host memory.
        hipblasPointerMode_t desiredPointerMode = HIPBLAS_POINTER_MODE_HOST;
        hipblasPointerMode_t pointerMode;
        HBCHECK(hipblasGetPointerMode(this->libContext.GetHandle(), &pointerMode));
        if(pointerMode != desiredPointerMode)
        {
            std::cout << "Changing pointer mode to read scalars from host memory." << std::endl;
            HBCHECK(hipblasSetPointerMode(this->libContext.GetHandle(), desiredPointerMode));
        }
#endif // READY

        auto func = axpy<ScalarType>();
        HBCHECK(func(this->libContext.GetHandle(),
                            n,
                            alpha.GetDeviceData(),
                            x.GetDeviceData(),
                            x.GetIncrement(),
                            y.GetDeviceData(),
                            y.GetIncrement()));

        y.CopyDeviceToHostAsync(this->hipStream);

        this->hipStream.Synchronize();
    }

    void Check(ScalarType relErrTolerance) const override
    {
        auto a = alpha.El();
        for(auto i = 0; i < n; ++i)
        {
            auto expVal = (a + 2)*i;
            REQUIRE_THAT(y.El(i), Catch::Matchers::WithinRel(expVal, relErrTolerance));
        }
    }

    // Declare a Catch2 section for a test.
    static void TestSection(HipStream& hipStream)
    {
        using TesterType = AxpyTester<ScalarType>;

        SECTION(opname<ScalarType>)
        {
            // Specify the problem.
            int n = GENERATE(take(1, random(50, 150)));
            ScalarType alpha = GENERATE(take(1, random(-2.5, 2.5)));
            int incx = GENERATE(1, 4);
            int incy = GENERATE(1, 7);

            // Build a test driver.
            TesterType tester(n, alpha, incx, incy, hipStream);
            REQUIRE_NOTHROW(tester.Init());

            // Do the operation.
            REQUIRE_NOTHROW(tester.DoOperation());

            // Verify the result.
            ScalarType relErrTolerance = 0.0001;
            tester.Check(relErrTolerance);
        }
    }
};

} // namespace

