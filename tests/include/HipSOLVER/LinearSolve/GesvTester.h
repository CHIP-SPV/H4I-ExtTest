// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <cstddef>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "Catch2Session.h"
#include "HipSOLVER/HipsolverTester.h"
#include "Matrix.h"
#include "Vector.h"
#include "Scalar.h"


namespace H4I::ExtTest
{

// A class to test the LAPACK GESV linear solve operation.
// GESV computes A*X = B where A is an n x n GE matrix,
// and X and B are n x nrhs matrices.
//
// The test initializes elements of A and B to random values.
// It checks the result by multiplying A and the solution X,
// and testing the relative error of each element in AX
// with the corresponding element in B.
//
template<typename ScalarType>
class GesvTester : public HipsolverTester<ScalarType>
{
private:
    int n;
    int nrhs;
    Matrix<ScalarType> A;
    Vector<int> devIpiv;
    Matrix<ScalarType> B;
    Matrix<ScalarType> X;
    int niters;
    Scalar<int> devInfo;

    
    // Only allow types for specializations we provide.
    template<typename T>
    static std::string opname(void) = delete;

    template<typename T>
    static auto gesv_bufferSize(void) = delete;

    template<typename T>
    static auto gesv(void) = delete;

    // Specializations for float.
    template<>
    static std::string opname<float>(void) { return "sgesv"; }

    template<>
    static auto gesv_bufferSize<float>(void)
    {
        return hipsolverSSgesv_bufferSize;
    }

    template<>
    static auto gesv<float>(void)
    {
        return hipsolverSSgesv;
    }

    // Specializations for double.
    template<>
    static std::string opname<double>(void) { return "dgesv"; }

    template<>
    static auto gesv_bufferSize<double>(void)
    {
        return hipsolverDDgesv_bufferSize;
    }

    template<>
    static auto gesv<double>(void)
    {
        return hipsolverDDgesv;
    }

public:
    GesvTester(int _n,
                int _nrhs,
                HipStream& _hipStream)
      : HipsolverTester<ScalarType>(_hipStream),
        n(_n),
        nrhs(_nrhs),
        A(n, n),
        devIpiv(n),
        B(n, nrhs),
        X(n, nrhs)
    { }

    // Create the input with known values.  See the comment above
    // the class for the description of the input values and
    // expected result.
    void Init(void) override
    {
        // Generate some random input using the Catch2 generator.
        // In the following, we create a 1D collection of random values,
        // then bulk copy it onto the 2D Matrix's host buffer.  This
        // assumes that the Matrix storage is contiguous.
        // A "safer" way of doing this would be in doubly-nested
        // loop and using the Matrix class' El() method so that it
        // could handle any non-contiguous storage underlying the matrix.
        // Since we're using random values, it doesn't really matter if
        // the matrices are using row-major or column-major storage.
        auto avals = GENERATE_COPY(chunk(n*n, take(n*n, random<ScalarType>(-100.0, 100.0))));
        std::memcpy(A.GetHostData(), avals.data(), n*n*sizeof(ScalarType));
        A.CopyHostToDeviceAsync(this->hipStream);

        auto bvals = GENERATE_COPY(chunk(n*nrhs, take(n*nrhs, random<ScalarType>(-100.0, 100.0))));
        std::memcpy(B.GetHostData(), bvals.data(), n*nrhs*sizeof(ScalarType));
        B.CopyHostToDeviceAsync(this->hipStream);
        
        // We do not initialize the output matrix X.

        this->hipStream.Synchronize();
    }


    void DoOperation(void) override
    {
        // Determine the work buffer size.
        size_t worksize;
        auto bsfunc = gesv_bufferSize<ScalarType>();
        HSCHECK(bsfunc(this->libContext.GetHandle(),
                            n,
                            nrhs,
                            A.GetDeviceData(),
                            n,  // lda
                            devIpiv.GetDeviceData(),
                            B.GetDeviceData(),
                            n,  // ldb
                            X.GetDeviceData(),
                            n,  // ldx
                            &worksize));

        // Allocate work space.
        // No need to initialize it.
        // TODO do we have alignment concerns here?
        H4I::ExtTest::Buffer<std::byte>  workspace(worksize);

        // Do the actual call.
        auto opfunc = gesv<ScalarType>();
        HSCHECK(opfunc(this->libContext.GetHandle(),
                            n,
                            nrhs,
                            A.GetDeviceData(),
                            n,  // lda
                            devIpiv.GetDeviceData(),
                            B.GetDeviceData(),
                            n,  // ldb
                            X.GetDeviceData(),
                            n,  // ldx
                            workspace.GetDeviceData(),
                            worksize,
                            &niters,
                            devInfo.GetDeviceData()));

        // Check whether the operation succeeded.
        devInfo.CopyDeviceToHost();

        if(devInfo.El() == 0)
        {
            // The operation succeeded.

            // Copy the result and pivot vector back to host memory.
            X.CopyDeviceToHostAsync(this->hipStream);
            devIpiv.CopyDeviceToHostAsync(this->hipStream);
        }
        else
        {
            // TODO how do we indicate that the operation failed?
        }

        this->hipStream.Synchronize();
    }

    void Check(void) const override
    {
        // Multiply A and X.
        // We do this "by hand" on the CPU to avoid the dependence
        // on another BLAS library's GEMM.
        Matrix<ScalarType> AX(n, nrhs);
        for(auto r = 0; r < n; ++r)
        {
            for(auto c = 0; c < nrhs; ++c)
            {
                auto sum = 0.0;
                for(auto i = 0; i < n; ++i)
                {
                    sum += (A.El(r, i) * X.El(i, c));
                }
                AX.El(r, c) = sum;
            }
        }

        // Check AX element-by-element against B.
        for(auto r = 0; r < n; ++r)
        {
            for(auto c = 0; c < nrhs; ++c)
            {
                auto expVal = B.El(r, c);
                auto compVal = AX.El(r, c);
                REQUIRE_THAT(compVal, Catch::Matchers::WithinRel(expVal, Catch2Session::theSession->GetRelErrThreshold<ScalarType>()));
            }
        }
    }

    // Declare a Catch2 section for a test.
    static void TestSection(HipStream& hipStream)
    {
        using TesterType = GesvTester<ScalarType>;

        SECTION(opname<ScalarType>())
        {
            // Specify the problem.
            int n = GENERATE(take(1, random(50, 150)));
            int nrhs = GENERATE(1, random(2, 20));

            // Build a test driver.
            TesterType tester(n, nrhs, hipStream);
            REQUIRE_NOTHROW(tester.Init());

            // Do the operation.
            REQUIRE_NOTHROW(tester.DoOperation());

            // Verify the result.
            tester.Check();
        }
    }
};

} // namespace

