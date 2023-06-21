// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#ifndef TEST_HIPSTREAM_H
#define TEST_HIPSTREAM_H

#include "hip/hip_runtime_api.h"
#include "HipstarException.h"

class HipStream
{
private:
    hipStream_t handle;

public:
    HipStream(void)
    {
        hipStreamCreate(&handle);        
    }

    ~HipStream(void)
    {
        CHECK(hipStreamDestroy(handle));
    }

    hipStream_t GetHandle(void) const   { return handle; }

    void Synchronize(void) const  { CHECK(hipStreamSynchronize(handle)); }
};

#endif // TEST_HIPSTREAM_H
