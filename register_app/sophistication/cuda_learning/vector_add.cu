
extern "C" __global__ void VectorAddKernel(const float* input_1, 
                                const float* input_2,
                                float* output,
                                unsigned int ElementNum)
{   

    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx < ElementNum){
        output[idx] = input_1[idx] + input_2[idx];
    }
}
