// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "hip/hip_runtime_api.h"
#include "HipstarLibraryContext.h"
#include "hipblas.h"
#include "HipStream.h"
#include "HipblasException.h"

class HipblasContext : public HipstarLibraryContext<hipblasHandle_t>
{
public:
    HipblasContext(void) = delete;

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
        handle = nullptr;
    }
};

