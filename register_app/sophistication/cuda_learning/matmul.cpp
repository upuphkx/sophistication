#include "matmul.cuh"

namespace test{
namespace ElementWiseOp{

CUDA_ELEMENTWISE_OP_RETURN_TYPE MatMul::ElementWiseImpl(){
    Enqueue();
    return CUDA_ELEMENTWISE_SUCCESS;
}

CUDA_FUNCTION_CALL_ERROR MatMul::RegisterInputOutput( ElementType* input_1,
                     ElementType* input_2,
                     ElementType* output,
                     uint32_t rows_1,
                     uint32_t columns,
                     uint32_t rows_2)
{
    input_1_ = input_1;
    input_2_ = input_2;
    output_ = output;
    rows_1_ = rows_1;
    rows_2_ = rows_2;
    columns_ = columns;
    return kCUDA_CALL_SUCCESS;
}                    

CUDA_FUNCTION_CALL_ERROR MatMul::GetOutput(){
    for (int i = 0 ; i < rows_1_ * columns_ ; i++){    
        Log::LogMessage(output_[i]);
    }
    return kCUDA_CALL_SUCCESS;
}   

CUDA_FUNCTION_CALL_ERROR MatMul::Enqueue(){
    callMatMulKernel(input_1_, input_2_, output_, rows_1_, columns_, rows_2_);
    return kCUDA_CALL_SUCCESS;
}



REGISTER(MatMul);

} // ElementWiseOp
} // test