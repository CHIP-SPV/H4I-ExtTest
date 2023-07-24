// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.

#include "Catch2Session.h"

using SessionType = H4I::ExtTest::Catch2Session;

// The singleton Catch2 Session object.
std::unique_ptr<SessionType> SessionType::theSession = std::make_unique<SessionType>();

int
main(int argc, char* argv[])
{
    // Parse the command line.
    auto ret = SessionType::theSession->applyCommandLine(argc, argv);
    if(ret == 0)
    {
        // It's a valid command line.
        // Run the tests.
        ret = SessionType::theSession->run();
    }
    return ret;
}

