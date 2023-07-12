// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "HipBLAS/HipblasTester.h"
#include "Matrix.h"


template<typename ScalarType>
class GemmTester : public HipblasTester<ScalarType>
{
public:
    using ResultType = Matrix<ScalarType>;
    using OpStatusType = hipblasStatus_t;

private:
    // Our matrices.
    Matrix<ScalarType> A;   // m x n
    Matrix<ScalarType> B;   // n x k, may be transposed
    Matrix<ScalarType> C;   // m x k
    ScalarType alpha;
    ScalarType beta;

    bool transB;

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
        // Generic version should never be called.
        // Specializations provided later.
        assert(false);
    }

public:
    GemmTester(int _m,
                int _n,
                int _k,
                ScalarType _alpha,
                ScalarType _beta,
                bool _transB,
                HipStream& _hipStream)
      : HipblasTester<ScalarType>(_hipStream),
        A(_m, _k),
        B( _transB ? _n : _k, _transB ? _k : _n ),
        C(_m, _n),
        alpha(_alpha),
        beta(_beta),
        transB(_transB)
    { }

    // Create the input matrices with known values.
    // Current test is:
    // * Items in col 0 of A are all 1, otherwise 0.
    // * Items in logical row 0 of B are all 1, otherwise 0.
    // * Storage for B in memory may be transposed.
    // * C[r, c] = r*c.
    // After the GEMM, C[r,c] should be alpha + beta * r * c.
    void Init(void) override
    {
        for(auto r = 0; r < A.GetNumRows(); ++r)
        {
            A.El(r, 0) = 1;
        }
        A.CopyHostToDeviceAsync(this->hipStream);

        for(auto c = 0; c < (transB ? B.GetNumRows() : B.GetNumCols()); ++c)
        {
            auto val = 1;
            if(transB)
            {
                B.El(c, 0) = val;
            }
            else
            {
                B.El(0, c) = val;
            }
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
        // Use of nRows does not differ depending on whether B is transposed.
        HBCHECK(CallGemm(this->blasContext.GetHandle(),
                            HIPBLAS_OP_N,
                            transB ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                            A.GetNumRows(),
                            C.GetNumCols(),
                            A.GetNumCols(),
                            &alpha,
                            A.GetDeviceData(),
                            A.GetNumRows(),
                            B.GetDeviceData(),
                            B.GetNumRows(),
                            &beta,
                            C.GetDeviceData(),
                            C.GetNumRows()));
        this->hipStream.Synchronize();

        // Read computed result from device to host.
        C.CopyDeviceToHostAsync(this->hipStream);
    }

    bool Check(ScalarType relErrTolerance) const override
    {
        uint32_t nMismatches = 0;

        auto expNumRows = A.GetNumRows();
        auto expNumCols = transB ? B.GetNumRows() : B.GetNumCols();

        // Verify result has right shape.
        // If not, all elements mismatch.
        if((C.GetNumRows() == expNumRows) and (C.GetNumCols() == expNumCols))
        {
            // Result has correct shape.
            // Check error in each element is below given tolerance.
            // This assumes column major ordering.
            for(auto c = 0; c < C.GetNumCols(); ++c)
            {
                for(auto r = 0; r < C.GetNumRows(); ++r)
                {
                    auto expVal = (alpha + beta * r * c);
                    auto compVal = C.El(r, c);
                    auto err = HipTester<ScalarType>::RelativeError(expVal, compVal);
                    if(err > relErrTolerance)
                    {
                        ++nMismatches;
                        std::cout << "mismatch at: (" << r << ", " << c << ")"
                            << " expected " << expVal
                            << ", got " << compVal
                            << ", rerr: " << err
                            << std::endl;
                    }
                }
            }
        }
        else
        {
            // Result does not have correct shape.
            // Consider all elements as mismatches.
            nMismatches = expNumRows * expNumCols;
        }
        return (nMismatches == 0);
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


