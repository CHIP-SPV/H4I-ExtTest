// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <tuple>
#include "Buffer.h"

namespace H4I::ExtTest
{

// A Matrix in CPU and GPU memory.
// The matrix elements are stored in column major order
// to be easier to pass to traditional BLAS library
// implementations that were originally designed for
// Fortran applications.
template<typename T>
class Matrix : public Buffer
{
protected:
    int nRows;
    int nCols;

public:
    Matrix(void) = delete;
    Matrix(int _nRows, int _nCols)
      : Buffer(_nRows * _nCols * sizeof(T)),
        nRows(_nRows),
        nCols(_nCols)
    { }

    Matrix(const std::pair<int, int>& dims)
      : Buffer(dims.first * dims.second * sizeof(T)),
        nRows(dims.first),
        nCols(dims.second)
    { }


    int GetNumRows(void) const   { return nRows; }
    int GetNumCols(void) const   { return nCols; }
    int GetNumItems(void) const  { return nRows * nCols; }

    T* GetDeviceData(void) const  { return reinterpret_cast<T*>(devData); }
    T* GetHostData(void) const { return reinterpret_cast<T*>(hostData); }

    // Access element from host storage.
    T& El(int r, int c)
    {
        return GetHostData()[c*nRows + r];
    }

    const T& El(int r, int c) const
    {
        return GetHostData()[c*nRows + r];
    }
};

} // namespace

