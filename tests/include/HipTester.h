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

public:
    HipTester(void) = delete;

    virtual ~HipTester(void) { }

    virtual void Init(void) = 0;

    virtual void DoOperation(void) = 0;

    virtual void Check(ScalarType relErrTolerance) const = 0;
};

