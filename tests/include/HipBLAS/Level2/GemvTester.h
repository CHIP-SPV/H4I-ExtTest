// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "Catch2Session.h"
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
//
// Note: unlike Level3's GEMM, specifying transA == true
// does not change the dimensions of A.  As per the
// Netlib description of GEMV, it does change the
// lengths of x and y.
//
// After the operation,
//   y[i] should be (using LaTeX representation):
//
//   if we did not transpose A:
//     y is of length m with:
//     y[i] = 2*\beta*i + \alpha*\sum_{k=0}^{m-1}k
//          = 2*\beta*i + \alpha*\sum_{k=1}^{m-1}k
//          = 2*\beta*i + \alpha*[m*(m-1)]/2
//
//   if we did transpose A:
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

    // Disallow types for which we don't specialize.
    template<typename T>
    inline static std::string opname;

    template<typename T>
    static auto gemv(void) = delete;

    // Specialize for float.
    template<>
    inline static const std::string opname<float> = "sgemv";

    template<>
    static auto gemv<float>(void)   { return hipblasSgemv; }

    // Specialize for double.
    template<>
    inline static const std::string opname<double> = "dgemv";

    template<>
    static auto gemv<double>(void)   { return hipblasDgemv; }

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
        HBCHECK(hipblasGetPointerMode(this->libContext.GetHandle(), &pointerMode));
        if(pointerMode != desiredPointerMode)
        {
            std::cout << "Changing pointer mode to read scalars from host memory." << std::endl;
            HBCHECK(hipblasSetPointerMode(this->libContext.GetHandle(), desiredPointerMode));
        }
#endif // READY

        // This assumes column major ordering of A (the use of nRows for leading dimension).
        auto func = gemv<ScalarType>();
        HBCHECK(func(this->libContext.GetHandle(),
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

    void Check(void) const override
    {
        auto xlen = transA ? A.GetNumRows() : A.GetNumCols();
        auto ylen = transA ? A.GetNumCols() : A.GetNumRows();

        REQUIRE(x.GetNumItems() == xlen);
        REQUIRE(y.GetNumItems() == ylen);

        for(auto i = 0; i < ylen; ++i)
        {
            ScalarType expVal = alpha*(xlen*(xlen-1))/2.0 + 2*beta*i;
            REQUIRE_THAT(y.El(i), Catch::Matchers::WithinRel(expVal, Catch2Session::theSession->GetRelErrThreshold<ScalarType>()));
        }
    }

    // Declare a Catch2 section for a test.
    static void TestSection(HipStream& hipStream)
    {
        using TesterType = GemvTester<ScalarType>;

        SECTION(opname<ScalarType>)
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
            tester.Check();
        }
    }

};

} // namespace

