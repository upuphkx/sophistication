#include "test.h"
using namespace test::factory;

void testMatMul(){
    Log::LogMessage<std::string>("[TEST CUDA MatMul PROGRAMING BEGINING]");
    test::ElementWiseOp::ElementWiseOp* cuda_matmul_api = static_cast<test::ElementWiseOp::ElementWiseOp*>(Factory::get_instance("MatMul")());
    // cuda_matmul_api->RuntimeEnqueue();
    float A[] = {1,1,1,1,1,1,1,1,1};
    std::vector<uint32_t> s {3, 3};
    Shape shape(s);
    Tensor input_1(&shape, A);
    CHECK_RETURN_STATUS(cuda_matmul_api->RegisterInputOutput(&input_1, &input_1, &input_1), "cuda_api matmul register input output");
    CHECK_RETURN_STATUS(cuda_matmul_api->GetOutput(), "cuda_api get output");
    CHECK_RETURN_STATUS(cuda_matmul_api->ElementWiseImpl(), "cuda_api call impl");
    CHECK_RETURN_STATUS(cuda_matmul_api->GetOutput(), "cuda_api get output");
    SAFE_DELETE_PTR(cuda_matmul_api);
    Log::LogMessage<std::string>("[TEST CUDA MatMul PROGRAMING ENDING]");
}