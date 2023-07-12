// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "hipblas.h"
#include "HipBLAS/HipblasException.h"
#include "HipBLAS/HipblasContext.h"
#include "HipTester.h"


template<typename ScalarType>
class HipblasTester : public HipTester<ScalarType>
{
protected:
    HipblasContext blasContext;

    HipblasTester(HipStream& _hipStream)
      : HipTester<ScalarType>(_hipStream),
        blasContext(this->hipStream)
    { }

public:
    HipblasTester(void) = delete;
};

