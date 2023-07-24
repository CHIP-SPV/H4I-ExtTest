// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <sstream>
#include <memory>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>


namespace H4I::ExtTest {

class Catch2Session : public Catch::Session
{
private:
    float relErrThreshold = 0.0001;

public:
    Catch2Session(void)
    {
        // Add our custom option to the Catch2 command line parser.
        std::ostringstream relErrOptDescStream;
        relErrOptDescStream << "relative error threshold (default: " << relErrThreshold << ")";

        auto mycli = cli() |
            Catch::Clara::Opt(relErrThreshold, "relerr")["--relError"](relErrOptDescStream.str());
        cli(mycli);
    }

    template<typename T>
    T GetRelErrThreshold(void) const = delete;

    template<>
    float GetRelErrThreshold<float>(void) const { return relErrThreshold; }

    template<>
    double GetRelErrThreshold<double>(void) const { return relErrThreshold; }

    static std::unique_ptr<Catch2Session> theSession;
};

} // namespace

