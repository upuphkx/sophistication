#include "internal_include/cufft_api_test.h"

namespace test{
namespace cufft_api{

TESTCUFFTAPIRETURNSTATUS cufft1DC2CTest::cufftAPIImpl(const uint32_t& n,
                                         const uint32_t& batch_size){
    // Log::LogMessage<std::string>("[TSET START] cufft_api: cufft1DC2C");
    cufftHandle plan;
    // cudaStream_t stream = NULL;
    uint32_t fft_size = batch_size * n;
    std::vector<data_type> data(fft_size);
    // Log::LogMessage<std::string>("Input array:");
    for (int32_t i = 0 ; i < fft_size ; i++){
        data[i] = data_type(i ,-i);
        // Log::LogMessage<data_type>(data[i]);
    }
    cufftComplex* d_data = nullptr;
    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlan1d(&plan, data.size(), CUFFT_C2C, batch_size));

    // CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // CUFFT_CALL(cufftSetStream(plan, stream));
    // create device data array
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(data_type) * data.size()));
    CUDA_RT_CALL(cudaMemcpyAsync(d_data, data.data(), sizeof(data_type) * data.size(), cudaMemcpyHostToDevice, NULL));
    
    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    // CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));

    CUDA_RT_CALL(cudaMemcpyAsync(data.data(), d_data, sizeof(data_type) * data.size(), cudaMemcpyDeviceToHost, NULL));
    // CUDA_RT_CALL(cudaStreamSynchronize(stream));

    // Log::LogMessage<std::string>("Output array:");
    // for (auto &i : data){
    //     Log::LogMessage<data_type>(i);
    // } 
    // free 
    CUDA_RT_CALL(cudaFree(d_data));
    CUFFT_CALL(cufftDestroy(plan));
    // CUDA_RT_CALL(cudaStreamDestroy(stream));
    CUDA_RT_CALL(cudaDeviceReset());


    // Log::LogMessage<std::string>("[TSET END] cufft_api: cufft1DC2C");
    return TEST_CUFFT_SUCCESS;
}

REGISTER(cufft1DC2CTest);
} // cufft_api
} // test