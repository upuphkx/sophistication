#include "vectoradd.cuh"

__global__ void VectorAddKernel(ElementType* input_1, 
                                ElementType* input_2,
                                ElementType* output,
                                uint32_t ElementNum)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx < ElementNum){
        output[idx] = input_1[idx] + input_2[idx];
    }
}

void CallVectorAddKernelFunction(std::vector<ElementType> input_1, 
                                std::vector<ElementType> input_2, 
                                ElementType* output,
                                uint32_t ElementNum)
{   
    ElementType* d_input_1 = NULL;
    ElementType* d_input_2 = NULL;
    ElementType* d_output = NULL;
    cudaMalloc(reinterpret_cast<void**>(&d_input_1), sizeof(ElementType) * ElementNum);
    cudaMalloc(reinterpret_cast<void**>(&d_input_2), sizeof(ElementType) * ElementNum);
    cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(ElementType) * ElementNum);
    
    cudaMemcpy(d_input_1, input_1.data(), sizeof(ElementType) * ElementNum, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_2, input_2.data(), sizeof(ElementType) * ElementNum, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, sizeof(ElementType) * ElementNum, cudaMemcpyHostToDevice);
 
    VectorAddKernel<<<1, 10>>>(d_input_1, d_input_2, d_output, ElementNum);

    cudaMemcpy(output, d_output, sizeof(ElementType) * ElementNum, cudaMemcpyDeviceToHost);
    cudaFree(d_input_1);
    cudaFree(d_input_2);
    cudaFree(d_output);
}
