#include "test.h"
using namespace test::factory;

void testMatMul(){
    Log::LogMessage<std::string>("[TEST CUDA MatMul PROGRAMING BEGINING]");
    test::ElementWiseOp::ElementWiseOp* cuda_matmul_api = static_cast<test::ElementWiseOp::ElementWiseOp*>(Factory::get_instance("MatMul")());
    cuda_matmul_api->RuntimeEnqueue();
    float A[] = {1,1,1,1,1,1,1,1,1};
    CHECK_RETURN_STATUS(cuda_matmul_api->RegisterInputOutput(A, A, A, 3, 3, 3), "cuda_api matmul register input output");
    CHECK_RETURN_STATUS(cuda_matmul_api->GetOutput(), "cuda_api get output");
    CHECK_RETURN_STATUS(cuda_matmul_api->ElementWiseImpl(), "cuda_api call impl");
    CHECK_RETURN_STATUS(cuda_matmul_api->GetOutput(), "cuda_api get output");
    SAFE_DELETE_PTR(cuda_matmul_api);
    Log::LogMessage<std::string>("[TEST CUDA MatMul PROGRAMING ENDING]");
}