// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#ifndef HIPBLAS_EXCEPTION_H
#define HIPBLAS_EXCEPTION_H

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

#endif // HIPBLAS_EXCEPTION_H
