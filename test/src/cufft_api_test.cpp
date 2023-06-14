#include "test.h"

void testCUFFT(){
    Log::LogMessage<std::string>("[TEST CUFFT API BEGINING]");
    std::vector<std::string> cufft_api_list {"cufft1DC2CTest", 
                                             "cufft1DR2CTest",
                                             "cufft1DMGPUC2CTest",
                                             "cufft2DC2RTest",
                                             "cufft3DC2CTest",
                                             "cufft3DMGPUC2CTest",
                                             "cufft3DMGPUR2CC2RTest"};
    for (std::string i_cufft_api : cufft_api_list) {
        test::cufft_api::CufftAPITest* cufft_api = static_cast<test::cufft_api::CufftAPITest*>(test::factory::Factory::get_instance(i_cufft_api)());
        CHECK_RETURN_STATUS(cufft_api->cufftAPIImpl(8, 1), i_cufft_api);
        SAFE_DELETE_PTR(cufft_api);
    }
    Log::LogMessage<std::string>("[TEST CUFFT API ENDING]");
}