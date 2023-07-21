// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "Buffer.h"

namespace H4I::ExtTest
{

// A scalar in CPU and GPU memory.
// NB: we could use a vector of size 1 for this.
template<typename T>
class Scalar : public Buffer<T>
{
public:
    Scalar(T initialHostVal = 0)
      : Buffer<T>(1)
    {
        El() = initialHostVal;
    }

    // Access element from host storage.
    T& El(void)
    {
        return this->GetHostData()[0];
    }

    const T& El(void) const
    {
        return this->GetHostData()[0];
    }
};

} // namespace

