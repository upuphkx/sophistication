#include "cufft_api_test.h"

namespace test{
namespace cufft_api{
TESTCUFFTAPIRETURNSTATUS cufft3DMGPUR2CC2RTest::cufftAPIImpl(const uint32_t& n,
                                     const uint32_t& batch_size)
{
    return TEST_CUFFT_NOT_IMPLEMENT;
}
REGISTER(cufft3DMGPUR2CC2RTest);
}
}