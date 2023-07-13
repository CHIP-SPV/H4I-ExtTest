// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "HipBLAS/HipblasTester.h"
#include "Matrix.h"
#include "Vector.h"

namespace H4I::ExtTest
{

// A tester for the BLAS *GEMV operation.
// Current test is:
// * A is m x n.  All elements of matrix A are 1.
// * x[i] = i with increment incx
// * y[i] = 2*i with increment incy
// After the operation,
//   y[i] should be (using LaTeX representation):
//
//   if we did the test with A not transposed:
//     y is of length m with:
//     y[i] = 2*\beta*i + \alpha*\sum_{k=0}^{m-1}k
//          = 2*\beta*i + \alpha*\sum_{k=1}^{m-1}k
//          = 2*\beta*i + \alpha*[m*(m-1)]/2
//
//   if we did the test with A transposed:
//     y is of length n with:
//     y[i] = 2*\beta*i + \alpha*[n*(n-1)]/2
//
template<typename ScalarType>
class GemvTester : public HipblasTester<ScalarType>
{
private:
    bool transA;
    Matrix<ScalarType> A;   // m x n
    Vector<ScalarType> x;   // if transA, length = m, else = n
    Vector<ScalarType> y;   // if transA, length = n, else = m
    ScalarType alpha;
    ScalarType beta;

    hipblasStatus_t CallGemv(hipblasHandle_t handle,
                                hipblasOperation_t trans,
                                int m,
                                int n,
                                const ScalarType* alpha,
                                const ScalarType* A,
                                int lda,
                                const ScalarType* x,
                                int incx,
                                const ScalarType* beta,
                                ScalarType* y,
                                int incy)
    {
        // Generic version should never be called.
        // Specializations provided later.
        assert(false);
    }

    static void TestSectionAux(std::string sectionName, HipStream& hipStream)
    {
        using TesterType = GemvTester<ScalarType>;

        SECTION(sectionName)
        {
            // Specify the problem.
            // A: m x n
            int m = GENERATE(take(1, random(50, 150)));
            int n = GENERATE(take(1, random(50, 150)));
            int incx = GENERATE(1, 4);
            int incy = GENERATE(1, 7);
            ScalarType alpha = GENERATE(take(1, random(-1.0, 1.0)));
            ScalarType beta = GENERATE(take(1, random(-2.5, 2.5)));
            auto transA = GENERATE(false, true);

            // Build a test driver.
            TesterType tester(transA, m, n, alpha, incx, beta, incy, hipStream);
            REQUIRE_NOTHROW(tester.Init());

            // Do the operation.
            REQUIRE_NOTHROW(tester.DoOperation());

            // Verify the result.
            ScalarType relErrTolerance = 0.0001;
            tester.Check(relErrTolerance);
        }
    }

public:
    GemvTester(bool _transA,
                int _m,
                int _n,
                ScalarType _alpha,
                int _incx,
                ScalarType _beta,
                int _incy,
                HipStream& _hipStream)
      : HipblasTester<ScalarType>(_hipStream),
        transA(_transA),
        A(_m, _n),
        alpha(_alpha),
        x(transA ? _m : _n, _incx),
        beta(_beta),
        y(transA ? _n : _m, _incy)
    { }

    // Create the input matrices with known values.
    void Init(void) override
    {
        // Assumes A is column major storage.
        for(auto c = 0; c < A.GetNumCols(); ++c)
        {
            for(auto r = 0; r < A.GetNumRows(); ++r)
            {
                A.El(r, c) = 1;
            }
        }
        A.CopyHostToDeviceAsync(this->hipStream);

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

        this->hipStream.Synchronize();
    }


    void DoOperation(void) override
    {
#if READY
        // We need the GEMV to assume our scalars
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

        // This assumes column major ordering of A (the use of nRows for leading dimension).
        HBCHECK(CallGemv(this->blasContext.GetHandle(),
                            transA ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                            A.GetNumRows(),
                            A.GetNumCols(),
                            &alpha,
                            A.GetDeviceData(),
                            A.GetNumRows(),
                            x.GetDeviceData(),
                            x.GetIncrement(),
                            &beta,
                            y.GetDeviceData(),
                            y.GetIncrement()));

        // Read computed result from device to host.
        y.CopyDeviceToHostAsync(this->hipStream);

        this->hipStream.Synchronize();
    }

    void Check(ScalarType relErrTolerance) const override
    {
        auto xlen = transA ? A.GetNumRows() : A.GetNumCols();
        auto ylen = transA ? A.GetNumCols() : A.GetNumRows();

        REQUIRE(x.GetNumItems() == xlen);
        REQUIRE(y.GetNumItems() == ylen);

        for(auto i = 0; i < ylen; ++i)
        {
            ScalarType expVal = alpha*(xlen*(xlen-1))/2.0 + 2*beta*i;
            REQUIRE_THAT(y.El(i), Catch::Matchers::WithinRel(expVal, relErrTolerance));
        }
    }

    // Declare a Catch2 section for a test.
    static void TestSection(HipStream& hipStream)
    {
        // Generic version should never be called.
        // Specializations provided later.
        assert(false);
    }
};


// Single-precision operation.
template<>
hipblasStatus_t
GemvTester<float>::CallGemv(hipblasHandle_t handle,
                            hipblasOperation_t trans,
                            int m,
                            int n,
                            const float* alpha,
                            const float* A,
                            int lda,
                            const float* x,
                            int incx,
                            const float* beta,
                            float* y,
                            int incy)
{
    return hipblasSgemv(handle,
            trans,
            m,
            n,
            alpha,
            A,
            lda,
            x,
            incx,
            beta,
            y,
            incy);
}


// Double-precision operation.
template<>
hipblasStatus_t
GemvTester<double>::CallGemv(hipblasHandle_t handle,
                            hipblasOperation_t trans,
                            int m,
                            int n,
                            const double* alpha,
                            const double* A,
                            int lda,
                            const double* x,
                            int incx,
                            const double* beta,
                            double* y,
                            int incy)
{
    return hipblasDgemv(handle,
            trans,
            m,
            n,
            alpha,
            A,
            lda,
            x,
            incx,
            beta,
            y,
            incy);
}


template<>
void
GemvTester<float>::TestSection(HipStream& hipStream)
{
    TestSectionAux("sgemv", hipStream);
}

template<>
void
GemvTester<double>::TestSection(HipStream& hipStream)
{
    TestSectionAux("dgemv", hipStream);
}

} // namespace

