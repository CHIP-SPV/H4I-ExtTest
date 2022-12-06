// Copyright 2021-2022 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#ifndef GEMMEX_TESTER_H
#define GEMMEX_TESTER_H

#include <iostream>
#include "HipStream.h"
#include "Matrix.h"

template<typename InType, typename OutType, bool Transpose = false>
class GemmExTester
{
public:
    // We'd like to expose OutType with the same name,
    // but C++ doesn't let us.
    using ComputeType = OutType;

protected:
    // Our matrices.
    // We define a D matrix even if the GEMM library we use doesn't use it.
    Matrix<InType> A;
    Matrix<InType> B;
    Matrix<OutType> C;
    Matrix<OutType> D;

    OutType alpha;
    OutType beta;

    const HipStream& hipStream;

    // Create the input matrices with known values.
    // Current test is:
    // * Items in col 0 of A are all 1.  Otherwise 0.
    // * Items in logical row 0 of B are all 1.  Otherwise 0.
    // * Storage for B in memory may be transposed.
    // * C[r, c] = r*c.
    // After the SGEMM, C[r,c] should be alpha + beta * r * c
    void InitMatrices(void)
    {
        for(auto r = 0; r < A.GetNumRows(); ++r)
        {
            A.El(r, 0) = 1;
        }
        A.CopyHostToDeviceAsync(hipStream);

        for(auto c = 0; c < (Transpose ? B.GetNumRows() : B.GetNumCols()); ++c)
        {
            auto val = 1;
            if(Transpose)
            {
                B.El(c, 0) = val;
            }
            else
            {
                B.El(0, c) = val;
            }
        }
        B.CopyHostToDeviceAsync(hipStream);

        for(auto c = 0; c < C.GetNumCols(); ++c)
        {
            for(auto r = 0; r < C.GetNumRows(); ++r)
            {
                C.El(r, c) = r*c;
            }
        }
        C.CopyHostToDeviceAsync(hipStream);

        // We don't need to initialize any values in D. 
        // Either it is only used as an output, or
        // it is not used by the GemmEx() implementation.
    }

    virtual bool UsesD(void) const = 0;

public:
    GemmExTester(int m,
                    int n,
                    int k,
                    OutType _alpha,
                    OutType _beta,
                    const HipStream& _hipStream)
      : A(m, k),
        B( Transpose ? n : k, Transpose ? k : n ),
        C(m, n),
        D(m, n),
        alpha(_alpha),
        beta(_beta),
        hipStream(_hipStream)
    {
        InitMatrices();
    }

    virtual ~GemmExTester(void)
    {
        // nothing to do.
    }

    void DumpTo(std::ostream& os) const
    {
        os << "alpha: " << alpha
            << "\nbeta: " << beta
            << "\nA: " << A
            << "\nB: " << B
            << "\nC: " << C
            << "\nD: " << D
            << std::endl;
    }

    virtual void DoGemmEx(void) = 0;
    
    void CheckComputation(void) const
    {
        auto& outputMatrix = this->UsesD() ? D : C;

        // Assumes column major ordering.
        uint32_t nMismatches = 0;
        for(auto c = 0; c < outputMatrix.GetNumCols(); c++)
        {
            for(auto r = 0; r < outputMatrix.GetNumRows(); r++)
            {
                auto expVal = (alpha + beta * r * c);
                auto compVal = outputMatrix.El(r,c);
                if(compVal != expVal)
                {
                    ++nMismatches;
                    std::cout << "mismatch at: (" << r << ", " << c << ")"
                        << " expected " << expVal
                        << ", got " << compVal
                        << std::endl;
                }
            }
        }
        std::cout << "Total mismatches: " << nMismatches << std::endl;
    }
};

template<typename InType, typename OutType, bool Transpose>
std::ostream&
operator<<(std::ostream& os, const GemmExTester<InType, OutType, Transpose>& tester)
{
    tester.DumpTo(os);
    return os;
}

#endif // GEMMEX_TESTER_H
