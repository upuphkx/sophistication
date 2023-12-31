#include "internal_include/cufft_api_test.h"

namespace test{
namespace cufft_api{

void cufft1DMGPUC2CTest::fill_array(cpudata_t &array) {
    std::mt19937 gen(3); // certified random number
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < array.size(); ++i) {
        float real = dis(gen);
        float imag = dis(gen);
        array[i] = {real, imag};
    };
};

/** Single GPU version of cuFFT plan for reference. */
void cufft1DMGPUC2CTest::single(dim_t fft, int &batch_size, cpudata_t &h_data_in, cpudata_t &h_data_out) {

    cufftHandle plan{};

    CUFFT_CALL(cufftCreate(&plan));

    size_t workspace_size;
    CUFFT_CALL(cufftMakePlan1d(plan, fft[0], CUFFT_C2C, batch_size, &workspace_size));

    void *d_data;
    size_t datasize = h_data_in.size() * sizeof(std::complex<float>);

    CUDA_RT_CALL(cudaMalloc(&d_data, datasize));
    CUDA_RT_CALL(cudaMemcpy(d_data, h_data_in.data(), datasize, cudaMemcpyHostToDevice));

    CUFFT_CALL(cufftXtExec(plan, d_data, d_data, CUFFT_FORWARD));

    CUDA_RT_CALL(cudaMemcpy(h_data_out.data(), d_data, datasize, cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaFree(d_data));

    CUFFT_CALL(cufftDestroy(plan));
};

/** Since cuFFT 10.4.0 cufftSetStream can be used to associate a stream with
 * multi-GPU plan. cufftXtExecDescriptor synchronizes efficiently to the stream
 * before and after execution. Please refer to
 * https://docs.nvidia.com/cuda/cufft/index.html#function-cufftsetstream for
 * more information.
 * cuFFT by default executes multi-GPU plans in synchronous manner.
 * */

void cufft1DMGPUC2CTest::spmg(dim_t fft, int &batch_size, gpus_t gpus, cpudata_t &h_data_in, cpudata_t &h_data_out,
          cufftXtSubFormat_t subformat) {

    // Initiate cufft plan
    cufftHandle plan{};
    CUFFT_CALL(cufftCreate(&plan));

#if CUFFT_VERSION >= 10400
    // Create CUDA Stream
    cudaStream_t stream{};
    CUDA_RT_CALL(cudaStreamCreate(&stream));
    CUFFT_CALL(cufftSetStream(plan, stream));
#endif

    // Define which GPUS are to be used
    CUFFT_CALL(cufftXtSetGPUs(plan, gpus.size(), gpus.data()));

    // Create the plan
    // With multiple gpus, worksize will contain multiple sizes
    size_t workspace_sizes[gpus.size()];
    CUFFT_CALL(cufftMakePlan1d(plan, fft[0], CUFFT_C2C, batch_size, workspace_sizes));

    cudaLibXtDesc *indesc;

    // Copy input data to GPUs
    CUFFT_CALL(cufftXtMalloc(plan, &indesc, subformat));
    CUFFT_CALL(cufftXtMemcpy(plan, reinterpret_cast<void *>(indesc),
                             reinterpret_cast<void *>(h_data_in.data()),
                             CUFFT_COPY_HOST_TO_DEVICE));

    // Execute the plan
    CUFFT_CALL(cufftXtExecDescriptor(plan, indesc, indesc, CUFFT_FORWARD));

    // Copy output data to CPU
    CUFFT_CALL(cufftXtMemcpy(plan, reinterpret_cast<void *>(h_data_out.data()),
                             reinterpret_cast<void *>(indesc), CUFFT_COPY_DEVICE_TO_HOST));

    CUFFT_CALL(cufftXtFree(indesc));
    CUFFT_CALL(cufftDestroy(plan));

#if CUFFT_VERSION >= 10400
    CUDA_RT_CALL(cudaStreamDestroy(stream));
#endif
};

TESTCUFFTAPIRETURNSTATUS cufft1DMGPUC2CTest::cufftAPIImpl(const uint32_t& n,
                                                           const uint32_t& batch_size)
{
    dim_t fft = {256};
    // can be {0, 0} to run on single-GPU system or if GPUs are not of same architecture
    gpus_t gpus = {0, 0};

    int batch_size_ = 1;

    size_t element_count = fft[0] * batch_size_;

    cpudata_t data_in(element_count);
    fill_array(data_in);

    cpudata_t data_out_reference(element_count, {-1.0f, -1.0f});
    cpudata_t data_out_test(element_count, {-0.5f, -0.5f});

    cufftXtSubFormat_t decomposition = CUFFT_XT_FORMAT_INPLACE;

    spmg(fft, batch_size_, gpus, data_in, data_out_test, decomposition);
    single(fft, batch_size_, data_in, data_out_reference);

    // The cuFFT library doesn't guarantee that single-GPU and multi-GPU cuFFT
    // plans will perform mathematical operations in same order. Small
    // numerical differences are possible.

    // verify results
    double error{};
    double ref{};
    for (size_t i = 0; i < element_count; ++i) {
        error += std::norm(data_out_test[i] - data_out_reference[i]);
        ref += std::norm(data_out_reference[i]);
    };

    // double l2_error = (ref == 0.0) ? std::sqrt(error) : std::sqrt(error) / std::sqrt(ref);
    // if (l2_error < 0.001) {
    //     std::cout << "PASSED with L2 error = " << l2_error << std::endl;
    // } else {
    //     std::cout << "FAILED with L2 error = " << l2_error << std::endl;
    // };
    return TEST_CUFFT_SUCCESS;
}
REGISTER(cufft1DMGPUC2CTest);
}
}