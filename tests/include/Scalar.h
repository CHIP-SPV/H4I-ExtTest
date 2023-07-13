// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "Buffer.h"

// A scalar in CPU and GPU memory.
// NB: we could use a vector of size 1 for this.
template<typename T>
class Scalar : public Buffer
{
public:
    Scalar(T initialHostVal = 0)
      : Buffer(sizeof(T))
    {
        El() = initialHostVal;
    }

    T* GetDeviceData(void) const  { return reinterpret_cast<T*>(devData); }
    T* GetHostData(void) const { return reinterpret_cast<T*>(hostData); }

    // Access element from host storage.
    T& El(void)
    {
        return GetHostData()[0];
    }

    const T& El(void) const
    {
        return GetHostData()[0];
    }
};

