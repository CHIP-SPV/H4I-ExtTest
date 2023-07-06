// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "hip/hip_runtime_api.h"
#include "hipblas.h"
#include "HipblasException.h"

class HipblasContext
{
private:
    hipblasHandle_t handle;

public:
    HipblasContext(void)
    {
        HBCHECK(hipblasCreate(&handle));

        // Use the default HIP stream.
        // (I.e., don't call hipblasSetStream.)
    }

    HipblasContext(const HipStream& stream)
    {
        HBCHECK(hipblasCreate(&handle));
        auto h = stream.GetHandle();
        if(h != nullptr)
        {
            HBCHECK(hipblasSetStream(handle, stream.GetHandle()));
        }
    }

    ~HipblasContext(void)
    {
        HBCHECK(hipblasDestroy(handle));
    }

    hipblasHandle_t GetHandle(void) const   { return handle; }
};

