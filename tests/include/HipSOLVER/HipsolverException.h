// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <string>
#include <sstream>
#include "HipstarException.h"

using HipsolverException = HipstarException<hipsolverStatus_t>;

inline
void
HSCHECK(hipsolverStatus_t code)
{
    if(code != HIPSOLVER_STATUS_SUCCESS)
    {
        std::ostringstream mstr;
        mstr << "hipSOLVER call failed, code: " << code;
        throw HipsolverException(code, mstr.str());
    }
}

