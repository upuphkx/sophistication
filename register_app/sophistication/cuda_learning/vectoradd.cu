#include "vectoradd.cuh"
template<typename T>
__global__ void VectorAddKernel(T* input_1, 
                                T* input_2,
                                T* output,
                                uint32_t ElementNum)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx < ElementNum){
        output[idx] = input_1[idx] + input_2[idx];
    }
}

template<typename DataType>
void kernelWrapper(Tensor* input_1, 
                                Tensor*  input_2, 
                                Tensor*  output)
{  
    DataType* d_input_1 = NULL;
    DataType* d_input_2 = NULL;
    DataType* d_output = NULL;
    cudaMalloc(reinterpret_cast<void**>(&d_input_1), input_1->getByte());
    cudaMalloc(reinterpret_cast<void**>(&d_input_2), input_2->getByte());
    cudaMalloc(reinterpret_cast<void**>(&d_output), output->getByte());
    
    cudaMemcpy(d_input_1, reinterpret_cast<DataType*>(input_1->getBuffer()), input_1->getByte(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_2, reinterpret_cast<DataType*>(input_2->getBuffer()), input_2->getByte(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, reinterpret_cast<DataType*>(output->getBuffer()), output->getByte(), cudaMemcpyHostToDevice);
 
    VectorAddKernel<DataType><<<1, 10>>>(d_input_1, d_input_2, d_output, input_1->getELementNum());

    cudaMemcpy(output->getBuffer(), reinterpret_cast<void*>(d_output), output->getByte(), cudaMemcpyDeviceToHost);
    cudaFree(d_input_1);
    cudaFree(d_input_2);
    cudaFree(d_output);
}    


void CallVectorAddKernelFunction(Tensor* input_1, 
                                Tensor*  input_2, 
                                Tensor*  output)
{   
    auto type = input_1->getDataType();
    switch (type){
        case DataType::kInt32_t:
            kernelWrapper<int32_t>(input_1, input_2, output);
            break;            
        case DataType::kInt64_t:
            kernelWrapper<long>(input_1, input_2, output);
            break;                        
        case DataType::kFloat32_t:
            kernelWrapper<float>(input_1, input_2, output);
            break;
        case DataType::kFloat64_t:
            kernelWrapper<double>(input_1, input_2, output);
            break;
    }
}