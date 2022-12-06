// Copyright 2021-2022 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#ifndef HIPBLAS_SGEMM_TESTER_H
#define HIPBLAS_SGEMM_TESTER_H

#include "hipblas.h"
#include "HipStream.h"
#include "HipblasException.h"
#include "SgemmTester.h"
#include "HipblasContext.h"

template<bool Transpose = false>
class HipblasSgemmTester : public SgemmTester<Transpose>
{
public:
    using ExceptionType = HipblasException;

protected:
    bool UsesD(void) const override { return false; }

public:
    HipblasSgemmTester(int m,
                        int n,
                        int k,
                        float alpha,
                        float beta,
                        const HipStream& hipStream)
      : SgemmTester<Transpose>(m, n, k, alpha, beta, hipStream)
    {
        // nothing else to do.
    }

    // Do the GEMM on the GPU.
    void
    DoSgemm(void) override
    {
        HipblasContext blasContext(this->hipStream);

#if READY
        // We need the GEMM to assume our scalars
        // are in host memory.
        hipblasPointerMode_t desiredPointerMode = HIPBLAS_POINTER_MODE_HOST;
        hipblasPointerMode_t pointerMode;
        CHECK(hipblasGetPointerMode(blasContext.GetHandle(), &pointerMode));
        if(pointerMode != desiredPointerMode)
        {
            std::cout << "Changing pointer mode to read scalars from host memory." << std::endl;
            CHECK(hipblasSetPointerMode(blasContext.GetHandle(), desiredPointerMode));
        }
#endif // rEADY

        // This assumes column major ordering (the use of nRows for leading dimension).
        // Use of nRows does not differ depending on whether B is transposed.
        CHECK(hipblasSgemm(blasContext.GetHandle(),
                            HIPBLAS_OP_N,
                            Transpose ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                            this->A.GetNumRows(),
                            this->C.GetNumCols(),
                            this->A.GetNumCols(),
                            &(this->alpha),
                            this->A.GetDeviceData(),
                            this->A.GetNumRows(),
                            this->B.GetDeviceData(),
                            this->B.GetNumRows(),
                            &(this->beta),
                            this->C.GetDeviceData(),
                            this->C.GetNumRows()));
        this->hipStream.Synchronize();

        // Read computed result from device to host.
        this->C.CopyDeviceToHostAsync(this->hipStream);
    }
};

#endif // HIPBLAS_SGEMM_TESTER_H
