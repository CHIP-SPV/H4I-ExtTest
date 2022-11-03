// Copyright 2021 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#ifndef TEST_HIPBLAS_CONTEXT_H
#define TEST_HIPBLAS_CONTEXT_H

#include "hip/hip_runtime_api.h"
#include "hipblas.h"
#include "HipblasException.h"

class HipblasContext
{
private:
    hipblasHandle_t handle;

public:
    HipblasContext(const HipStream& stream)
    {
        CHECK(hipblasCreate(&handle));
        CHECK(hipblasSetStream(handle, stream.GetHandle()));
    }

    ~HipblasContext(void)
    {
        // std::cerr << "In ~HipblasContext" << std::endl;
        CHECK(hipblasDestroy(handle));
    }

    hipblasHandle_t GetHandle(void) const   { return handle; }
};

#endif // TEST_HIPBLAS_CONTEXT_H
