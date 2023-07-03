#include <catch2/catch_test_macros.hpp>

#include <vector>
#include <tuple>

#include "HipBLAS/GemmTester.h"


// Declare a Catch2 section for a GEMM test.
template<typename ScalarType>
void
GemmTestSection(std::string sectionName, HipStream& hipStream)
{
    SECTION(sectionName)
    {
        // Specify the problem.
        // A: m x k
        // B: k x n
        // C: m x n
        int m = 100;
        int n = 50;
        int k = 150;
        ScalarType alpha = 0.5;
        ScalarType beta = 0.75;

        std::vector<std::pair<std::string, bool>> subsectionInfo {
            { "no trans B", false },
            { "trans B", true }
        };

        for(auto& x : subsectionInfo)
        {
            SECTION(x.first)
            {
                using TesterType = GemmTester<ScalarType>;

                // Build a test driver.
                TesterType tester(m, n, k, alpha, beta, x.second, hipStream);

                // Do the operation.
                typename TesterType::ResultType result(m, k);
                auto opStatus = tester.DoOperation(result);
                hipStream.Synchronize();
                REQUIRE(opStatus == HIPBLAS_STATUS_SUCCESS);

                // Verify the result.
                if(opStatus == HIPBLAS_STATUS_SUCCESS)
                {
                    auto nMismatches = tester.Check(result);
                    REQUIRE(nMismatches == 0);
                }
            }
        }
    }
}

TEST_CASE("GEMM", "[BLAS][BLAS3]")
{
    HipStream hipStream;

    GemmTestSection<float>("sgemm", hipStream);
    GemmTestSection<double>("dgemm", hipStream);
}

