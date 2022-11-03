// Copyright 2021 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#ifndef DO_MAIN_H
#define DO_MAIN_H

#include <iostream>
#include "CommandLine.h"
#include "HipStream.h"

template<typename TesterType>
int
DoMain(int argc, char* argv[])
{
    int ret = 0;

    try
    {
        bool shouldRun = true;

        // Variables that specify matrix sizes.
        // A: m x k
        // B: k x n
        // C: m x n
        int m = -1;
        int k = -1;
        int n = -1;

        // Scaling factors for A*B and for C as input.
        float alpha;
        float beta;

        // Whether we should dump debugging output.
        bool verbose;

        // Parse the command line.
        std::tie(shouldRun,
                    ret,
                    m,
                    k,
                    n,
                    alpha,
                    beta,
                    verbose) = ParseCommandLine<float>(argc, argv);

        if(shouldRun)
        {
            // Build a HIP stream.
            HipStream hipStream;

            // Create the input matrices with known values.
            TesterType tester(m, n, k, alpha, beta, hipStream);

            // Wait for matrices to be copied to GPU.
            hipStream.Synchronize();

            if(verbose)
            {
                // Dump the state of the problem on the GPU for debugging.
                std::cout << tester << std::endl;
            }

            // Do the GEMM.
            tester.DoSgemm();
            hipStream.Synchronize();

            if(verbose)
            {
                // Dump the state after the GEMM for debugging.
                std::cout << tester << std::endl;
            }

            // Verify the GPU-computed results match the expected results.
            tester.CheckComputation();
        }
    }
    catch(const HipException& e)
    {
        std::cerr << "In HipException catch block" << std::endl;
        std::cerr << "HIP Exception: " << e.GetCode() << ": " << e.what() << std::endl;
        ret = 1;
    }
    catch(const typename TesterType::ExceptionType& e)
    {
        std::cerr << "hipBLAS Exception: " << e.GetCode() << ": " << e.what() << std::endl;
        ret = 1;
    }
    catch(const std::exception& e)
    {
        std::cerr << "exception: " << e.what() << std::endl;
        ret = 1;
    }
    catch(...)
    {
        std::cerr << "unrecognized exception caught" << std::endl;
        ret = 1;
    }

    return ret;
}

#endif // DO_MAIN_H
