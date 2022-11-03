#include "HipblasSgemmTester.h"
#include "DoSgemmMain.h"

int
main(int argc, char* argv[])
{
    return DoMain<HipblasSgemmTester<true>>(argc, argv);
}

