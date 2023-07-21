// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "HipTester.h"

template<typename ScalarType, typename LibraryContextType>
class HipstarTester : public HipTester<ScalarType>
{
protected:
    LibraryContextType libContext;

    HipstarTester(HipStream& _hipStream)
      : HipTester<ScalarType>(_hipStream),
        libContext(this->hipStream)
    { }

public:
    HipstarTester(void) = delete;
};

