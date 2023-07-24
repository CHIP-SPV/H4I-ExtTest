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

// A class to test the BLAS Level 1 dot product.
// The input values are:
// x[i] = i with stride incx
// y[i] = 2*i with stride incy
//
// By the definition of the dot product,
// the result should be (in LaTeX representation)
//   \sum_{i=1}^{n-1} 2*i^2
// Using the fact that \sum_{i=1}^{n} i^2 == [n * (n+1) * (2*n + 1)] / 6,
// this simplifies to
//   (1/3) * (n-1) * n * (2*n - 1).
//
template<typename ScalarType>
class DotTester : public HipblasTester<ScalarType>
{
private:
    int n;
    Vector<ScalarType> x;
    Vector<ScalarType> y;
    Scalar<ScalarType> result;

    // Disallow types for which we don't specialize.
    template<typename T>
    inline static const std::string opname;

    template<typename T>
    static auto dot(void) = delete;

    // Specialize for float.
    template<>
    inline static const std::string opname<float> = "sdot";

    template<>
    static auto dot<float>(void)    { return hipblasSdot; }

    // Specialize for double.
    template<>
    inline static const std::string opname<double> = "ddot";

    template<>
    static auto dot<double>(void)    { return hipblasDdot; }

public:
    DotTester(int _n,
                int _incx,
                int _incy,
                HipStream& _hipStream)
      : HipblasTester<ScalarType>(_hipStream),
        n(_n),
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

        result.El() = 0;
        result.CopyHostToDeviceAsync(this->hipStream);

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

        auto func = dot<ScalarType>();
        HBCHECK(func(this->libContext.GetHandle(),
                            n,
                            x.GetDeviceData(),
                            x.GetIncrement(),
                            y.GetDeviceData(),
                            y.GetIncrement(),
                            result.GetDeviceData()));

        result.CopyDeviceToHostAsync(this->hipStream);

        this->hipStream.Synchronize();
    }

    void Check(void) const override
    {
        auto expVal = (static_cast<ScalarType>(1) / 3) * (n - 1) * n * (2*n - 1);
        REQUIRE_THAT(result.El(), Catch::Matchers::WithinRel(expVal, Catch2Session::theSession->GetRelErrThreshold<ScalarType>()));
    }

    // Declare a Catch2 section for a test.
    static void TestSection(HipStream& hipStream)
    {
        using TesterType = DotTester<ScalarType>;

        SECTION(opname<ScalarType>)
        {
            // Specify the problem.
            int n = GENERATE(take(1, random(50, 150)));
            int incx = GENERATE(1, 4);
            int incy = GENERATE(1, 7);

            // Build a test driver.
            TesterType tester(n, incx, incy, hipStream);
            REQUIRE_NOTHROW(tester.Init());

            // Do the operation.
            REQUIRE_NOTHROW(tester.DoOperation());

            // Verify the result.
            tester.Check();
        }
    }

};

} // namespace

