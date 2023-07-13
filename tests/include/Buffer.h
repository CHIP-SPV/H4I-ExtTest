// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <vector>
#include <cstring>  // for memset
#include "hip/hip_runtime.h"
#include "HipstarException.h"

namespace H4I::ExtTest
{

// A memory buffer in CPU and GPU memory,
// with operations to transfer contents
// between the memory spaces.
class Buffer
{
protected:
    size_t size;
    char* hostData;
    char* devData;

public:
    Buffer(size_t _size)
      : size(_size),
        hostData(nullptr),
        devData(nullptr)
    {
        HIPCHECK(hipHostMalloc(&hostData, size));
        memset(hostData, 0, size);

        HIPCHECK(hipMalloc(&devData, size));
        HIPCHECK(hipMemset(devData, 0, size));
    }

    ~Buffer(void)
    {
        if(hostData != nullptr)
        {
            HIPCHECK(hipHostFree(hostData));
            hostData = nullptr;
        }
        if(devData != nullptr)
        {
            HIPCHECK(hipFree(devData));
            devData = nullptr;
        }
    }

    size_t GetSize(void) const    { return size; }

    void CopyHostToDevice(void)
    {
        HIPCHECK(hipMemcpy(devData,
                        hostData,
                        size,
                        hipMemcpyHostToDevice));
    }

    void CopyHostToDeviceAsync(const HipStream& stream)
    {
        HIPCHECK(hipMemcpyAsync(devData,
                            hostData,
                            size,
                            hipMemcpyHostToDevice,
                            stream.GetHandle()));
    }

    void CopyDeviceToHost(void)
    {
        HIPCHECK(hipMemcpy(hostData,
                        devData,
                        size,
                        hipMemcpyDeviceToHost));
    }

    void CopyDeviceToHostAsync(const HipStream& stream)
    {
        HIPCHECK(hipMemcpyAsync(hostData,
                            devData,
                            size,
                            hipMemcpyDeviceToHost,
                            stream.GetHandle()));
    }
};

} // namespace

