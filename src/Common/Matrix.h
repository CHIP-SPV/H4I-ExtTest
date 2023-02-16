// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <cstring>  // for memset
#include "hip/hip_runtime.h"

#include "src/Common/ExtTestConfig.h"
#if defined(TEST_HALF_PRECISION)
#include "hip/hip_fp16.h"
#endif // defined(TEST_HALF_PRECISION)

#include "HipstarException.h"

// A Matrix in CPU and GPU memory.
// The matrix elements are stored in column major order
// to be easier to pass to traditional BLAS library
// implementations that were originally designed for
// Fortran applications.
template<typename T>
class Matrix
{
protected:
    int nRows;
    int nCols;

    T* hostData;
    T* devData;

public:
    Matrix(int _nRows, int _nCols)
      : nRows(_nRows),
        nCols(_nCols),
        hostData(nullptr),
        devData(nullptr)
    {
        CHECK(hipHostMalloc(&hostData, GetSize()));
        memset(hostData, 0, GetSize());

        CHECK(hipMalloc(&devData, GetSize()));
        CHECK(hipMemset(devData, 0, GetSize()));
    }

    ~Matrix(void)
    {
        if(hostData != nullptr)
        {
            CHECK(hipHostFree(hostData));
            hostData = nullptr;
        }
        if(devData != nullptr)
        {
            CHECK(hipFree(devData));
            devData = nullptr;
        }
    }

    int GetNumRows(void) const   { return nRows; }
    int GetNumCols(void) const   { return nCols; }
    int GetNumItems(void) const  { return nRows * nCols; }
    int GetSize(void) const    { return GetNumItems() * sizeof(T); }

    T* GetDeviceData(void) const  { return devData; }
    T* GetHostData(void) const { return hostData; }

    // Access element from host storage.
    T& El(int r, int c)
    {
        return hostData[c*nRows + r];
    }

    const T& El(int r, int c) const
    {
        return hostData[c*nRows + r];
    }

    void CopyHostToDevice(void)
    {
        CHECK(hipMemcpy(devData,
                        hostData,
                        GetSize(),
                        hipMemcpyHostToDevice));
    }

    void CopyHostToDeviceAsync(const HipStream& stream)
    {
        CHECK(hipMemcpyAsync(devData,
                            hostData,
                            GetSize(),
                            hipMemcpyHostToDevice,
                            stream.GetHandle()));
    }

    void CopyDeviceToHost(void)
    {
        CHECK(hipMemcpy(hostData,
                        devData,
                        GetSize(),
                        hipMemcpyDeviceToHost));
    }

    void CopyDeviceToHostAsync(const HipStream& stream)
    {
        CHECK(hipMemcpyAsync(hostData,
                            devData,
                            GetSize(),
                            hipMemcpyDeviceToHost,
                            stream.GetHandle()));
    }
};


#if defined(TEST_HALF_PRECISION)
inline
float
ToFloat(const __half& h)
{
    return __half2float(h);
}
#endif // defined(TEST_HALF_PRECISION)

inline
float
ToFloat(const float& f)
{
    return f;
}

// Dump a Matrix's data from device to the given stream.
// Does *not* change the Matrix's data in host memory.
template<typename T>
std::ostream&
operator<<(std::ostream& os, const Matrix<T>& m)
{
    std::vector<T> hdata(m.GetNumItems());
    auto matrixSize = m.GetSize();
    CHECK(hipMemcpy(hdata.data(), m.GetDeviceData(), matrixSize, hipMemcpyDeviceToHost));
    os << "dims: " << m.GetNumRows() << 'x' << m.GetNumCols()
        << ", nItems: " << m.GetNumItems()
        << ", size: " << matrixSize
        << ", vals: ";
    for(auto i = 0; i < m.GetNumItems(); ++i)
    {
        os << ToFloat(hdata[i]) << ' ';
    }
    return os;
}

#endif // MATRIX_H
