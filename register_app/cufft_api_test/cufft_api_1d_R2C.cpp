#include "cufft_api_test.h"

namespace test{
namespace cufft_api{

TESTCUFFTAPIRETURNSTATUS cufft1DR2CTest::cufftAPIImpl(const uint32_t& n, const uint32_t& batch_size){
    // Log::LogMessage<std::string>("[TSET START] cufft_api: cufft1DR2C");
    cufftHandle plan;
    cudaStream_t stream = NULL;
    using input_type = float32_t;
    uint32_t fft_size = batch_size * n;

    std::vector<input_type> input(fft_size);
    std::vector<data_type> output(static_cast<uint32_t>((fft_size / 2 + 1)));
    // Log::LogMessage<std::string>("Input array:");
    for (int i = 0 ; i < fft_size ; i++){
        input[i] = static_cast<input_type>(i);
        // Log::LogMessage<input_type>(input[i]);
    }

    input_type* d_input = nullptr;
    cufftComplex* d_output = nullptr;

    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlan1d(&plan, input.size(), CUFFT_R2C, batch_size));

    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan, stream));

    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(input_type) * input.size()));
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(data_type) * output.size()));

    CUDA_RT_CALL(cudaMemcpyAsync(d_input, input.data(), sizeof(input_type) * input.size(), cudaMemcpyHostToDevice, stream));

    CUFFT_CALL(cufftExecR2C(plan, d_input, d_output));

    CUDA_RT_CALL(cudaMemcpyAsync(output.data(), d_output, sizeof(data_type) * output.size(), cudaMemcpyDeviceToHost, stream));

    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    // Log::LogMessage<std::string>("Output array:");
    // for (int i = 0 ; i < output.size() ; i++){
    //     Log::LogMessage<data_type>(output[i]);
    // }

    CUDA_RT_CALL(cudaFree(d_input));
    CUDA_RT_CALL(cudaFree(d_output));

    CUFFT_CALL(cufftDestroy(plan));
    CUDA_RT_CALL(cudaStreamDestroy(stream));
    CUDA_RT_CALL(cudaDeviceReset());
    // Log::LogMessage<std::string>("[TSET END] cufft_api: cufft1DR2C");
    return TEST_CUFFT_SUCCESS;
}

REGISTER(cufft1DR2CTest);

} // cufft_api
} // test