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
    virtual CUDA_FUNCTION_CALL_ERROR RegisterInputOutput(std::vector<ElementType> input_1, 
                                                        std::vector<ElementType>  input_2, 
                                                        ElementType*  output,
                                                        uint32_t ElementNum) = 0;

    virtual CUDA_FUNCTION_CALL_ERROR RegisterInputOutput( ElementType* input_1,
                     ElementType* input_2,
                     ElementType* output,
                     uint32_t rows_1,
                     uint32_t columns,
                     uint32_t rows_2) = 0;      

    virtual CUDA_FUNCTION_CALL_ERROR GetOutput() = 0;

    virtual CUDA_FUNCTION_CALL_ERROR RuntimeEnqueue() = 0;
    
    virtual ~ElementWiseOp(){}
};

// template<class ElementType>
class VectorAdd : public ElementWiseOp
{
public:
    virtual CUDA_ELEMENTWISE_OP_RETURN_TYPE ElementWiseImpl() override final;

    virtual CUDA_FUNCTION_CALL_ERROR RegisterInputOutput(std::vector<ElementType> input_1, 
                                                        std::vector<ElementType>  input_2, 
                                                        ElementType*  output,
                                                        uint32_t ElementNum) override final;

    virtual CUDA_FUNCTION_CALL_ERROR RegisterInputOutput( ElementType* input_1,
                     ElementType* input_2,
                     ElementType* output,
                     uint32_t rows_1,
                     uint32_t columns,
                     uint32_t rows_2) override final {return kCUDA_CALL_INVALID;}

    virtual CUDA_FUNCTION_CALL_ERROR GetOutput() override final;

    CUDA_FUNCTION_CALL_ERROR Enqueue();

    virtual CUDA_FUNCTION_CALL_ERROR RuntimeEnqueue() override final;

    CUDA_FUNCTION_CALL_ERROR compileFileToBC(const std::string &cu_file_name, 
                                                    char** ptxResult, 
                                                    size_t* ptxResultSize);
    CUDA_FUNCTION_CALL_ERROR loadBC(char* ptx, CUmodule& module);

    virtual ~VectorAdd() override final {}


private:
    std::vector<ElementType> input_1_;
    std::vector<ElementType> input_2_;
    ElementType* output_;
    uint32_t ElementNum_;
};

class MatMul : public ElementWiseOp{
public:
    virtual CUDA_ELEMENTWISE_OP_RETURN_TYPE ElementWiseImpl() override final;

    virtual CUDA_FUNCTION_CALL_ERROR RegisterInputOutput(std::vector<ElementType> input_1, 
                                                        std::vector<ElementType>  input_2, 
                                                        ElementType*  output,
                                                        uint32_t ElementNum) override final
    {
        return kCUDA_CALL_INVALID;
    }

    virtual CUDA_FUNCTION_CALL_ERROR RegisterInputOutput( ElementType* input_1,
                     ElementType* input_2,
                     ElementType* output,
                     uint32_t rows_1,
                     uint32_t columns,
                     uint32_t rows_2) override final;

    virtual CUDA_FUNCTION_CALL_ERROR GetOutput() override final;      

    CUDA_FUNCTION_CALL_ERROR Enqueue();

    virtual CUDA_FUNCTION_CALL_ERROR RuntimeEnqueue() override final {return kCUDA_CALL_INVALID;}

    virtual ~MatMul() override final {}
private:
    // dim3 shape_;
    ElementType* input_1_;
    ElementType* input_2_;
    ElementType* output_;
    uint32_t rows_1_;
    uint32_t rows_2_;
    uint32_t columns_;
};
} // namespace ElementWiseOp
} // namespace test
#endif // __TEST_ELEMENT_OP_H