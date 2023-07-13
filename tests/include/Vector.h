// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "Buffer.h"

namespace H4I::ExtTest
{

// A Vector in CPU and GPU memory.
template<typename T>
class Vector : public Buffer
{
protected:
    int n;
    int stride; // TODO only strides >=1 are supported

public:
    Vector(int _n, int _stride = 1)
      : Buffer(_n * _stride * sizeof(T)),
        n(_n),
        stride(_stride)
    { }

    int GetNumItems(void) const  { return n; }
    int GetIncrement(void) const { return stride; }

    T* GetDeviceData(void) const  { return reinterpret_cast<T*>(devData); }
    T* GetHostData(void) const { return reinterpret_cast<T*>(hostData); }

    // Access element from host storage.
    T& El(int i)
    {
        return GetHostData()[stride * i];
    }

    const T& El(int i) const
    {
        return GetHostData()[stride * i];
    }
};

} // namespace

