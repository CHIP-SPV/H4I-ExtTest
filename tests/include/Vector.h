// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "Buffer.h"

namespace H4I::ExtTest
{

// A Vector in CPU and GPU memory.
template<typename T>
class Vector : public Buffer<T>
{
protected:
    int n;
    int stride; // TODO only strides >=1 are supported

public:
    Vector(int _n, int _stride = 1)
      : Buffer<T>(_n * _stride),
        n(_n),
        stride(_stride)
    { }

    int GetNumItems(void) const  { return n; }
    int GetIncrement(void) const { return stride; }

    // Access element from host storage.
    T& El(int i)
    {
        return this->GetHostData()[stride * i];
    }

    const T& El(int i) const
    {
        return this->GetHostData()[stride * i];
    }
};

} // namespace

