// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "HipBLAS/HipblasTester.h"
#include "Vector.h"
#include "Scalar.h"


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
    int incx;
    Vector<ScalarType> y;
    int incy;
    Scalar<ScalarType> result;

    hipblasStatus_t CallDot(hipblasHandle_t handle,
                                int n,
                                const ScalarType* x,
                                int incx,
                                const ScalarType* y,
                                int incy,
                                ScalarType* result)
    {
        // This generic version should never be called.
        // Specializations will be provided later.
        assert(false);
    }

    static void TestSectionAux(std::string sectionName, HipStream& hipStream)
    {
        using TesterType = DotTester<ScalarType>;

        SECTION(sectionName)
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
            ScalarType relErrTolerance = 0.0001;
            tester.Check(relErrTolerance);
        }
    }

public:
    DotTester(int _n,
                int _incx,
                int _incy,
                HipStream& _hipStream)
      : HipblasTester<ScalarType>(_hipStream),
        n(_n),
        x(_n, _incx),
        incx(_incx),
        y(_n, _incy),
        incy(_incy)
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
        HBCHECK(hipblasGetPointerMode(this->blasContext.GetHandle(), &pointerMode));
        if(pointerMode != desiredPointerMode)
        {
            std::cout << "Changing pointer mode to read scalars from host memory." << std::endl;
            HBCHECK(hipblasSetPointerMode(this->blasContext.GetHandle(), desiredPointerMode));
        }
#endif // READY

        // This assumes column major ordering (the use of nRows for leading dimension).
        // Use of nRows does not differ depending on whether B is transposed.
        auto ret = CallDot(this->blasContext.GetHandle(),
                            n,
                            x.GetDeviceData(),
                            incx,
                            y.GetDeviceData(),
                            incy,
                            result.GetDeviceData());
        HBCHECK(ret);

        result.CopyDeviceToHostAsync(this->hipStream);

        this->hipStream.Synchronize();
    }

    void Check(ScalarType relErrTolerance) const override
    {
        auto expVal = (static_cast<ScalarType>(1) / 3) * (n - 1) * n * (2*n - 1);  // TODO doesn't work if incx, incy != 1
        REQUIRE_THAT(result.El(), Catch::Matchers::WithinRel(expVal, relErrTolerance));
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
DotTester<float>::CallDot(hipblasHandle_t handle,
                            int n,
                            const float* x,
                            int incx,
                            const float* y,
                            int incy,
                            float* result)
{
    return hipblasSdot(handle,
            n,
            x,
            incx,
            y,
            incy,
            result);
}


// Double-precision operation.
template<>
hipblasStatus_t
DotTester<double>::CallDot(hipblasHandle_t handle,
                            int n,
                            const double* x,
                            int incx,
                            const double* y,
                            int incy,
                            double* result)
{
    return hipblasDdot(handle,
            n,
            x,
            incx,
            y,
            incy,
            result);
}


template<>
void
DotTester<float>::TestSection(HipStream& hipStream)
{
    TestSectionAux("sdot", hipStream);
}

template<>
void
DotTester<double>::TestSection(HipStream& hipStream)
{
    TestSectionAux("ddot", hipStream);
}

