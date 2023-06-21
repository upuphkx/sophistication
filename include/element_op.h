#ifndef __TEST_ELEMENT_OP_H
#define __TEST_ELEMENT_OP_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <fstream>
#include <streambuf>
#include <cstdint>
#include <stdio.h>
#include <vector>

#include "register.h"

typedef float ElementType;

namespace test {
namespace ElementWiseOp {

//CUDA API ERROR CHECKING
#define CUDA_RT_CALL(call)                                                \
    do {                                                                  \
        auto status = static_cast<cudaError_t>(call);                     \
        if(status != cudaSuccess)                                         \
            fprintf(stderr,                                               \
                "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "\
                "with "                                                   \
                "%s (%d).\n",                                             \
                #call,                                                    \
                __LINE__,                                                 \
                __FILE__,                                                 \
                cudaGetErrorString(status),                               \
                status);                                                  \
    } while(0)

enum CUDA_ELEMENTWISE_OP_RETURN_TYPE {
    CUDA_ELEMENTWISE_SUCCESS     = 1,
    CUDA_ELEMENTWISE_ERROR       = 2,
    CUDA_ELEMENTWISE_NOT_SUPPORT = 3,
    CUDA_ELEMENTWISE_INVALID     = 4
};

enum CUDA_FUNCTION_CALL_ERROR {
    kCUDA_CALL_SUCCESS  = 1,
    kCUDA_CALL_ERROR    = 0,
    kCUDA_CALL_INVALID  = 2
};

class ElementWiseOp 
{
public:
    virtual CUDA_ELEMENTWISE_OP_RETURN_TYPE ElementWiseImpl() = 0;
    
    virtual CUDA_FUNCTION_CALL_ERROR ElementWiseRuntimeEnqueueImpl() = 0;

    virtual CUDA_FUNCTION_CALL_ERROR RegisterInputOutput(Tensor* input_1, 
                                                        Tensor*  input_2, 
                                                        Tensor*  output
                                                        ) = 0; 

    virtual CUDA_FUNCTION_CALL_ERROR GetOutput() = 0;

    
    virtual ~ElementWiseOp(){}
};

// template<class ElementType>
class VectorAdd : public ElementWiseOp
{
public:
    CUDA_FUNCTION_CALL_ERROR Enqueue();
    
    virtual CUDA_ELEMENTWISE_OP_RETURN_TYPE ElementWiseImpl() override final;

    virtual CUDA_FUNCTION_CALL_ERROR RegisterInputOutput(Tensor* input_1, 
                                                        Tensor*  input_2, 
                                                        Tensor*  output) override final;

    virtual CUDA_FUNCTION_CALL_ERROR GetOutput() override final;


    virtual CUDA_FUNCTION_CALL_ERROR ElementWiseRuntimeEnqueueImpl() override final;

    CUDA_FUNCTION_CALL_ERROR compileFileToBC(const std::string &cu_file_name, 
                                                    char** ptxResult, 
                                                    size_t* ptxResultSize);
    CUDA_FUNCTION_CALL_ERROR loadBC(char* ptx, CUmodule& module);

    virtual ~VectorAdd() override final {}


private:
    Tensor*  input_1_;
    Tensor*  input_2_;
    Tensor*  output_;
};

class MatMul : public ElementWiseOp{
public:
    CUDA_FUNCTION_CALL_ERROR Enqueue();

    virtual CUDA_ELEMENTWISE_OP_RETURN_TYPE ElementWiseImpl() override final;

    virtual CUDA_FUNCTION_CALL_ERROR RegisterInputOutput(Tensor* input_1, 
                                                        Tensor*  input_2, 
                                                        Tensor*  output) override final;

    virtual CUDA_FUNCTION_CALL_ERROR GetOutput() override final;      


    virtual CUDA_FUNCTION_CALL_ERROR ElementWiseRuntimeEnqueueImpl() override final {return kCUDA_CALL_INVALID;}

    virtual ~MatMul() override final {}
private:
    Tensor*  input_1_;
    Tensor*  input_2_;
    Tensor*  output_;
};
} // namespace ElementWiseOp
} // namespace test
#endif // __TEST_ELEMENT_OP_H