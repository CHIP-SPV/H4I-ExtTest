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
class Matrix : public Buffer<T>
{
protected:
    int nRows;
    int nCols;

public:
    Matrix(void) = delete;
    Matrix(int _nRows, int _nCols)
      : Buffer<T>(_nRows * _nCols),
        nRows(_nRows),
        nCols(_nCols)
    { }

    Matrix(const std::pair<int, int>& dims)
      : Matrix(dims.first, dims.second)
    { }


    int GetNumRows(void) const   { return nRows; }
    int GetNumCols(void) const   { return nCols; }
    int GetNumItems(void) const  { return nRows * nCols; }

    // Access element from host storage.
    T& El(int r, int c)
    {
        return this->GetHostData()[c*nRows + r];
    }
    T& El(const std::pair<int, int>& idx)
    {
        return El(idx.first, idx.second);
    }

    const T& El(int r, int c) const
    {
        return this->GetHostData()[c*nRows + r];
    }
    const T& El(const std::pair<int, int>& idx) const
    {
        return El(idx.first, idx.second);
    }
};

} // namespace

