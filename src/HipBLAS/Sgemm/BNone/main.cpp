// Copyright 2021-2022 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include "HipblasSgemmTester.h"
#include "DoSgemmMain.h"

int
main(int argc, char* argv[])
{
    return DoMain<HipblasSgemmTester<false>>(argc, argv);
}

