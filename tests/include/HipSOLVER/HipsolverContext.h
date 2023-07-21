// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "hip/hip_runtime_api.h"
#include "HipstarLibraryContext.h"
#include "hipsolver.h"
#include "HipStream.h"
#include "HipsolverException.h"

class HipsolverContext : public HipstarLibraryContext<hipsolverHandle_t>
{
public:
    HipsolverContext(void) = delete;

    HipsolverContext(const HipStream& stream)
    {
        HSCHECK(hipsolverCreate(&handle));
        auto h = stream.GetHandle();
        if(h != nullptr)
        {
            HSCHECK(hipsolverSetStream(handle, stream.GetHandle()));
        }
    }

    ~HipsolverContext(void)
    {
        HSCHECK(hipsolverDestroy(handle));
        handle = nullptr;
    }
};

