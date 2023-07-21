// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <string>
#include <sstream>
#include "HipstarException.h"

using HipblasException = HipstarException<hipblasStatus_t>;

inline
void
HBCHECK(hipblasStatus_t code)
{
    if(code != HIPBLAS_STATUS_SUCCESS)
    {
        std::ostringstream mstr;
        mstr << "hipBLAS call failed, code: " << code;
        throw HipblasException(code, mstr.str());
    }
}

