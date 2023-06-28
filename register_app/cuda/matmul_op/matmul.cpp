#include "matmul.cuh"

namespace test{
namespace ElementWiseOp{

CUDA_ELEMENTWISE_OP_RETURN_TYPE MatMul::ElementWiseImpl(){
    Enqueue();
    return CUDA_ELEMENTWISE_SUCCESS;
}

CUDA_FUNCTION_CALL_ERROR MatMul::RegisterInputOutput(Tensor* input_1, 
                                                    Tensor*  input_2, 
                                                    Tensor*  output)
{
    input_1_ = input_1;
    input_2_ = input_2;
    output_ = output;
    return kCUDA_CALL_SUCCESS;
}                    

CUDA_FUNCTION_CALL_ERROR MatMul::GetOutput(){
    using type = float;
    uint32_t element_num = this->output_->getELementNum();
    for (uint32_t i = 0 ; i < element_num ; i++){    
        Log::LogMessage(reinterpret_cast<type*>(output_->getBuffer())[i]);
    }
    return kCUDA_CALL_SUCCESS;
}   

CUDA_FUNCTION_CALL_ERROR MatMul::Enqueue(){
    callMatMulKernel(input_1_, input_2_, output_);
    return kCUDA_CALL_SUCCESS;
}



REGISTER(MatMul);

} // ElementWiseOp
} // test