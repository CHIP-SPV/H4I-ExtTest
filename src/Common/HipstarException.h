// Copyright 2021-2022 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#ifndef HIPSTAR_EXCEPTION_H
#define HIPSTAR_EXCEPTION_H

#include <stdexcept>
#include <string>

template<typename ECodeType>
class HipstarException : public std::runtime_error
{
private:
    ECodeType code;

public:
    HipstarException(ECodeType _code, const char* _msg)
      : std::runtime_error(_msg),
        code(_code)
    { }

    HipstarException(ECodeType _code, const std::string& _msg)
      : std::runtime_error(_msg),
        code(_code)
    { }

    ECodeType GetCode(void) const { return code; }
};

using HipException = HipstarException<hipError_t>;


inline
void
CHECK(hipError_t code)
{
    if(code != hipSuccess)
    {
        throw HipException(code, "HIP call failed");
    }
}

#endif // HIPSTAR_EXCEPTION_H
