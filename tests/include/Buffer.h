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
template<typename T>
class Buffer
{
private:
    size_t size;    // number of T's
    size_t rawSize;
    T* hostData;
    T* devData;

public:
    Buffer(size_t _size)
      : size(_size),
        rawSize(_size * sizeof(T)),
        hostData(nullptr),
        devData(nullptr)
    {
        HIPCHECK(hipHostMalloc(&hostData, rawSize));
        memset(hostData, 0, rawSize);

        HIPCHECK(hipMalloc(&devData, rawSize));
        HIPCHECK(hipMemset(devData, 0, rawSize));
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

#if READY
    T* GetHostData(void)    { return hostData; }
    const T* const GetHostData(void) const  { return hostData; }

    T* GetDeviceData(void)    { return devData; }
    const T* const GetDeviceData(void) const  { return devData; }
#else
    T* GetHostData(void) const  { return hostData; }
    T* GetDeviceData(void) const    { return devData; }
#endif // READY

    void CopyHostToDevice(void)
    {
        HIPCHECK(hipMemcpy(devData,
                        hostData,
                        rawSize,
                        hipMemcpyHostToDevice));
    }

    void CopyHostToDeviceAsync(const HipStream& stream)
    {
        HIPCHECK(hipMemcpyAsync(devData,
                            hostData,
                            rawSize,
                            hipMemcpyHostToDevice,
                            stream.GetHandle()));
    }

    void CopyDeviceToHost(void)
    {
        HIPCHECK(hipMemcpy(hostData,
                        devData,
                        rawSize,
                        hipMemcpyDeviceToHost));
    }

    void CopyDeviceToHostAsync(const HipStream& stream)
    {
        HIPCHECK(hipMemcpyAsync(hostData,
                            devData,
                            rawSize,
                            hipMemcpyDeviceToHost,
                            stream.GetHandle()));
    }
};

} // namespace

