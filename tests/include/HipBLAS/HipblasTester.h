// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "HipstarTester.h"
#include "HipBLAS/HipblasContext.h"

template<typename ScalarType>
using HipblasTester = HipstarTester<ScalarType, HipblasContext>;

