// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "HipStream.h"
#include "HipstarException.h"


template<typename ScalarType>
class HipTester
{
protected:
    HipStream& hipStream;

    HipTester(HipStream& _hipStream)
      : hipStream(_hipStream)
    { }

    // Compute relative error of computed value compared to expected value.
    // NB: this is not *quite* the true relative error.  If the expected value
    // is 0, we return absolute error.
    static ScalarType RelativeError(ScalarType expVal, ScalarType compVal)
    {
        auto delta = compVal - expVal;
        return std::abs((expVal != 0) ? (delta / expVal) : delta);
    }

public:
    HipTester(void) = delete;

    virtual ~HipTester(void) { }

    virtual void Init(void) = 0;

    virtual void DoOperation(void) = 0;

    virtual bool Check(ScalarType relErrTolerance) const = 0;
};

