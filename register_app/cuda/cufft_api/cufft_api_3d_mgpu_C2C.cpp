#include "internal_include/cufft_api_test.h"

namespace test{
namespace cufft_api{

TESTCUFFTAPIRETURNSTATUS cufft3DMGPUC2CTest::cufftAPIImpl(const uint32_t& n,
                                     const uint32_t& batch_size)
{
    return TEST_CUFFT_NOT_IMPLEMENT;
}
REGISTER(cufft3DMGPUC2CTest);
}
}