#include "HipblasSgemmTester.h"
#include "DoSgemmMain.h"

int
main(int argc, char* argv[])
{
    return DoMain<HipblasSgemmTester<false>>(argc, argv);
}

