#include "test.h"
using namespace test::factory;
#define ELEMENT_NUM 5

template<typename T, int MAX_LENGTH>
T* initBuffer(){
    T* buffer = new T[MAX_LENGTH];
    for (int i = 0 ; i < MAX_LENGTH ; i++){
        buffer[i] = i*1.1;
    } 
    return buffer;
}

void testVectorAdd(){
    Log::LogMessage<std::string>("[TEST CUDA PROGRAMING BEGINING]");
    test::ElementWiseOp::ElementWiseOp* cuda_api = static_cast<test::ElementWiseOp::ElementWiseOp*>(Factory::get_instance("VectorAdd")());
    // cuda_api->ElementWiseRuntimeEnqueueImpl();
    std::vector<uint32_t> s {ELEMENT_NUM};
    Shape shape(s);
    float* A = initBuffer<float, ELEMENT_NUM>();
    float* B = initBuffer<float, ELEMENT_NUM>();
    float* C = initBuffer<float, ELEMENT_NUM>();

    Tensor input_1(&shape, reinterpret_cast<void*>(A));
    Tensor input_2(&shape, reinterpret_cast<void*>(B));
    Tensor output(&shape, reinterpret_cast<void*>(C));
    CHECK_RETURN_STATUS(cuda_api->RegisterInputOutput(input_1.setDataType(DataType::kFloat32_t), 
                                                        input_2.setDataType(DataType::kFloat32_t), 
                                                        output.setDataType(DataType::kFloat32_t)), "cuda_api register input output");
    CHECK_RETURN_STATUS(cuda_api->GetOutput(), "cuda_api get output");
    CHECK_RETURN_STATUS(cuda_api->ElementWiseImpl(), "cuda_api call impl");
    CHECK_RETURN_STATUS(cuda_api->GetOutput(), "cuda_api get output");
    SAFE_DELETE_PTR(cuda_api);
    delete [] A;
    delete [] B;
    delete [] C;
    Log::LogMessage<std::string>("[TEST CUDA PROGRAMING ENDING]");
}