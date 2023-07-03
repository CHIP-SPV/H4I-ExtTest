// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <iostream>

#include "hip/hip_runtime_api.h"
#include "HipstarException.h"

class HipStream
{
private:
    hipStream_t handle;

public:
    HipStream(void)
    {
        HIPCHECK(hipStreamCreate(&handle));
    }

    ~HipStream(void)
    {
        HIPCHECK(hipStreamDestroy(handle));
    }

    hipStream_t GetHandle(void) const   { return handle; }

    void Synchronize(void) const  { HIPCHECK(hipStreamSynchronize(handle)); }
};

