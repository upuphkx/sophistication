#include "vectoradd.cuh"

namespace test {
namespace ElementWiseOp {

CUDA_ELEMENTWISE_OP_RETURN_TYPE VectorAdd::ElementWiseImpl(){
    Enqueue();
    return CUDA_ELEMENTWISE_SUCCESS;
}

CUDA_FUNCTION_CALL_ERROR VectorAdd::RegisterInputOutput(Tensor* input_1, 
                                                        Tensor*  input_2, 
                                                        Tensor*  output)
{
    input_1_ = input_1;
    input_2_ = input_2;
    output_ = output;
    return kCUDA_CALL_SUCCESS;
}

CUDA_FUNCTION_CALL_ERROR VectorAdd::GetOutput(){
    Log::LogMessage(this->output_->getELementNum());
    float* buffer = reinterpret_cast<float*>(this->output_->getBuffer());
    for (int i = 0 ; i < this->output_->getELementNum() ; i++){
        Log::LogMessage(buffer[i]);
    }
    return kCUDA_CALL_SUCCESS;
}

CUDA_FUNCTION_CALL_ERROR VectorAdd::Enqueue(){
    CallVectorAddKernelFunction(this->input_1_, this->input_2_, this->output_);   
    return kCUDA_CALL_SUCCESS;
}

CUDA_FUNCTION_CALL_ERROR VectorAdd::compileFileToBC(const std::string &cu_file_name, 
                                                    char** ptxResult, 
                                                    size_t* ptxResultSize){
    std::ifstream file(cu_file_name.c_str());
    if (!file.is_open()){
        printf("\nerror: unable to open %s for reading!\n", cu_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    std::string file_stream((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    // Log(file_stream);
    nvrtcProgram prog;
    if(NVRTC_SUCCESS != nvrtcCreateProgram(&prog, file_stream.c_str(), cu_file_name.c_str(), 0, NULL, NULL)) {
            printf("\nerror:nvrtcCreateProgram failed\n");
            exit(EXIT_FAILURE);
        }
    int numCompileOptions = 0;
    char* compileParams = nullptr;
    if (NVRTC_SUCCESS != nvrtcCompileProgram(prog, numCompileOptions, &compileParams)) {
            printf("\nerror:nvrtcCompileProgram failed\n");
            exit(EXIT_FAILURE);
    }

    size_t ptx_size{};
    nvrtcGetPTXSize(prog, &ptx_size);

    char* ptx = reinterpret_cast<char*>(malloc(sizeof(char) * ptx_size));

    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);
    *ptxResult = ptx;
    *ptxResultSize = ptx_size;
    return kCUDA_CALL_SUCCESS;
}

CUDA_FUNCTION_CALL_ERROR VectorAdd::loadBC(char* ptx, CUmodule& module){
    cuInit(0);

    CUdevice device;
    cuDeviceGet(&device, 0);

    CUcontext context;
    cuCtxCreate(&context, 0, device);

    cuModuleLoadDataEx(&module, ptx, 0, 0, 0);

    return kCUDA_CALL_SUCCESS;
}

CUDA_FUNCTION_CALL_ERROR VectorAdd::ElementWiseRuntimeEnqueueImpl(){
    std::string cu_file_name = "./vector_add.cu";
    char* ptxResult;
    size_t ptxResultSize;
    compileFileToBC(cu_file_name, &ptxResult, &ptxResultSize);
    CUmodule module;
    loadBC(ptxResult, module);

    CUfunction kernel_addr;
    cuModuleGetFunction(&kernel_addr, module, "VectorAddKernel");

    float input_1[] = {1, 2, 3, 4};
    float input_2[] = {1, 2, 3, 4};
    float output[] = {0, 1, 0, 0};       

    float* A = NULL;
    float* B = NULL;
    float* result = NULL;
    unsigned int element_num = 4;
    void* arr[] = { reinterpret_cast<void*>(&A),
                reinterpret_cast<void*>(&B),
                reinterpret_cast<void*>(&result),
                /* guess if not specify it for element_num 
                    which is auto set a very large number */
                reinterpret_cast<void*>(&element_num) 
    };
    cudaMalloc(reinterpret_cast<void**>(&A), sizeof(float) * element_num);
    cudaMalloc(reinterpret_cast<void**>(&B), sizeof(float) * element_num);
    cudaMalloc(reinterpret_cast<void**>(&result), sizeof(float) * element_num);      

    cudaMemcpy(A, input_1, sizeof(float) * element_num, cudaMemcpyHostToDevice);
    cudaMemcpy(B, input_2, sizeof(float) * element_num, cudaMemcpyHostToDevice);
    cudaMemcpy(result, output, sizeof(float) * element_num, cudaMemcpyHostToDevice);
    int blockDimsX = 256;
    int gridDimsX = (element_num + blockDimsX - 1) / blockDimsX; 
    dim3 blockDims (blockDimsX, 1, 1);
    dim3 gridDims (gridDimsX, 1, 1);

    CUDA_RT_CALL(cuLaunchKernel(kernel_addr, 
                            gridDims.x, 
                            gridDims.y,
                            gridDims.z,
                            blockDims.x,
                            blockDims.y,
                            blockDims.z, 
                            0, 0, arr, 0));

    cudaMemcpy(output, result, sizeof(float) * element_num, cudaMemcpyDeviceToHost);
    for (int i = 0 ; i < 4 ; i++){
        Log::LogMessage(output[i]);
    }
    CUDA_RT_CALL(cudaFree(A));
    CUDA_RT_CALL(cudaFree(B));
    CUDA_RT_CALL(cudaFree(result));
    return kCUDA_CALL_SUCCESS;
}
REGISTER(VectorAdd);

} // namespace ElementWiseOp
} // namespace test