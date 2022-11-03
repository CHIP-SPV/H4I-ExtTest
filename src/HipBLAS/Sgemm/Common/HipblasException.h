// Copyright 2021 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#ifndef HIPBLAS_EXCEPTION_H
#define HIPBLAS_EXCEPTION_H

#include "HipstarException.h"

using HipblasException = HipstarException<hipblasStatus_t>;

inline
void
CHECK(hipblasStatus_t code)
{
    if(code != HIPBLAS_STATUS_SUCCESS)
    {
        throw HipblasException(code, "hipBLAS call failed");
    }
}

#endif // HIPBLAS_EXCEPTION_H
