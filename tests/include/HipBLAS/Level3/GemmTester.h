// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "HipBLAS/HipblasTester.h"
#include "Matrix.h"

namespace H4I::ExtTest
{

// Tester for GEMM operation.
// GEMM computes C = alpha * op(A) * op(B) + beta * C
// where:
//   A, B, and C are matrices
//   alpha and beta are scalars
//   for matrix M, op(M) means either M or M^T
//
// Since A and B may be transposed independently, there
// are four cases for op(A) * op(B).
//
// Since C is an m x n matrix always, each product must result
// in an m x n matrix.  Thus, A is m x k if not transposed, k x m if it is.
// And B is k x n if not transposed, n x k if it is.
// 
// To make the test tractable, we initialize op(A) and op(B) so that
// op(A) * op(B) is an m x n matrix where each element is 1.
// I.e., we initialize op(A) with 1s in the first column and 0s elsewhere,
// and op(B) with 1s in the first row and 0s elsewhere.
//
// We initialize C[r, c] to r*c.
//
// With this initial data, after the operation:
//   C[r, c] = alpha + beta * r * c
//
template<typename ScalarType>
class GemmTester : public HipblasTester<ScalarType>
{
private:
    bool transA;
    bool transB;
    int m;
    int n;
    int k;
    Matrix<ScalarType> A;
    Matrix<ScalarType> B;
    Matrix<ScalarType> C;
    ScalarType alpha;
    ScalarType beta;


    hipblasStatus_t CallGemm(hipblasHandle_t handle,
                                hipblasOperation_t transA,
                                hipblasOperation_t transB,
                                int m,
                                int n,
                                int k,
                                const ScalarType* alpha,
                                const ScalarType* A,
                                int lda,
                                const ScalarType* B,
                                int ldb,
                                const ScalarType* beta,
                                ScalarType* C,
                                int ldc)
    {
        // This generic version should never be called.
        // Specializations for specific types are provided later.
        assert(false);
    }

    static void TestSectionAux(std::string sectionName, HipStream& hipStream)
    {
        using TesterType = GemmTester<ScalarType>;

        SECTION(sectionName)
        {
            // Specify the problem.
            auto transA = true; // GENERATE(false, true);
            auto transB = false; // GENERATE(false, true);
            int m = GENERATE(take(1, random(50, 150)));
            int n = GENERATE(take(1, random(50, 150)));
            int k = GENERATE(take(1, random(50, 150)));
            ScalarType alpha = GENERATE(take(1, random(-1.0, 1.0)));
            ScalarType beta = GENERATE(take(1, random(-2.5, 2.5)));

            // Build a test driver.
            TesterType tester(transA, transB, m, n, k, alpha, beta, hipStream);
            REQUIRE_NOTHROW(tester.Init());

            // Do the operation.
            REQUIRE_NOTHROW(tester.DoOperation());

            // Verify the result.
            ScalarType relErrTolerance = 0.0001;
            tester.Check(relErrTolerance);
        }
    }

public:
    GemmTester(bool _transA,
                bool _transB,
                int _m,
                int _n,
                int _k,
                ScalarType _alpha,
                ScalarType _beta,
                HipStream& _hipStream)
      : HipblasTester<ScalarType>(_hipStream),
        transA(_transA),
        transB(_transB),
        m(_m),
        n(_n),
        k(_k),
        A(!transA ? std::pair(m, k) : std::pair(k, m)),
        B(!transB ? std::pair(k, n) : std::pair(n, k)),
        C(m, n),
        alpha(_alpha),
        beta(_beta)
    { }

    // Create the input matrices with known values as described
    // in the class comment.
    void Init(void) override
    {
        for(auto r = 0; r < m; ++r)
        {
            A.El(!transA ? std::pair(r, 0) : std::pair(0, r)) = 1;
        }
        A.CopyHostToDeviceAsync(this->hipStream);

        for(auto c = 0; c < n; ++c)
        {
            B.El(!transB ? std::pair(0, c) : std::pair(c, 0)) = 1;
        }
        B.CopyHostToDeviceAsync(this->hipStream);

        for(auto c = 0; c < C.GetNumCols(); ++c)
        {
            for(auto r = 0; r < C.GetNumRows(); ++r)
            {
                C.El(r, c) = r*c;
            }
        }
        C.CopyHostToDeviceAsync(this->hipStream);

        this->hipStream.Synchronize();
    }


    void DoOperation(void) override
    {
#if READY
        // We need the GEMM to assume our scalars
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
        // Use of nRows as lda, ldb does not differ depending on whether A, B are transposed.
        HBCHECK(CallGemm(this->blasContext.GetHandle(),
                            !transA ? HIPBLAS_OP_N : HIPBLAS_OP_T,
                            !transB ? HIPBLAS_OP_N : HIPBLAS_OP_T,
                            A.GetNumRows(),
                            C.GetNumCols(),
                            A.GetNumCols(),
                            &alpha,
                            A.GetDeviceData(),
                            !transA ? m : k,
                            B.GetDeviceData(),
                            !transB ? k : n,
                            &beta,
                            C.GetDeviceData(),
                            m));

        // Read computed result from device to host.
        C.CopyDeviceToHostAsync(this->hipStream);

        this->hipStream.Synchronize();
    }

    void Check(ScalarType relErrTolerance) const override
    {
        // Verify result has right shape.
        REQUIRE(C.GetNumRows() == m);
        REQUIRE(C.GetNumCols() == n);

        // Check error in each element is below given tolerance.
        // This assumes column major ordering.
        for(auto c = 0; c < C.GetNumCols(); ++c)
        {
            for(auto r = 0; r < C.GetNumRows(); ++r)
            {
                auto expVal = (alpha + beta * r * c);
                auto compVal = C.El(r, c);
                REQUIRE_THAT(compVal, Catch::Matchers::WithinRel(expVal, relErrTolerance));
            }
        }
    }

    // Declare a Catch2 section for a GEMM test.
    static void TestSection(HipStream& hipStream)
    {
        // Generic version should never be called.
        // Specializations provided later.
        assert(false);
    }
};


// Single-precision GEMM.
template<>
hipblasStatus_t
GemmTester<float>::CallGemm(hipblasHandle_t handle,
                            hipblasOperation_t transA,
                            hipblasOperation_t transB,
                            int m,
                            int n,
                            int k,
                            const float* alpha,
                            const float* A,
                            int lda,
                            const float* B,
                            int ldb,
                            const float* beta,
                            float* C,
                            int ldc)
{
    return hipblasSgemm(handle,
            transA,
            transB,
            m,
            n,
            k,
            alpha,
            A,
            lda,
            B,
            ldb,
            beta,
            C,
            ldc);
}


// Double-precision GEMM.
template<>
hipblasStatus_t
GemmTester<double>::CallGemm(hipblasHandle_t handle,
                            hipblasOperation_t transA,
                            hipblasOperation_t transB,
                            int m,
                            int n,
                            int k,
                            const double* alpha,
                            const double* A,
                            int lda,
                            const double* B,
                            int ldb,
                            const double* beta,
                            double* C,
                            int ldc)
{
    return hipblasDgemm(handle,
            transA,
            transB,
            m,
            n,
            k,
            alpha,
            A,
            lda,
            B,
            ldb,
            beta,
            C,
            ldc);
}


template<>
void
GemmTester<float>::TestSection(HipStream& hipStream)
{
    TestSectionAux("sgemm", hipStream);
}

template<>
void
GemmTester<double>::TestSection(HipStream& hipStream)
{
    TestSectionAux("dgemm", hipStream);
}

} // namespace

