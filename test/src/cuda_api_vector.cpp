#include "test.h"
using namespace test::factory;

void testCUDA(){
    Log::LogMessage<std::string>("[TEST CUDA PROGRAMING BEGINING]");
    test::ElementWiseOp::ElementWiseOp* cuda_api = static_cast<test::ElementWiseOp::ElementWiseOp*>(Factory::get_instance("VectorAdd")());
    cuda_api->RuntimeEnqueue();
    std::vector<float> A {1, 2, 3, 4};
    std::vector<float> B {1, 2, 3, 4};
    float C[] = {0, 0, 0, 0};
    CHECK_RETURN_STATUS(cuda_api->RegisterInputOutput(A, B, C, 4), "cuda_api register input output");
    CHECK_RETURN_STATUS(cuda_api->GetOutput(), "cuda_api get output");
    CHECK_RETURN_STATUS(cuda_api->ElementWiseImpl(), "cuda_api call impl");
    CHECK_RETURN_STATUS(cuda_api->GetOutput(), "cuda_api get output");
    SAFE_DELETE_PTR(cuda_api);
    Log::LogMessage<std::string>("[TEST CUDA PROGRAMING ENDING]");
}