#ifndef __TEST_CUDA_LEARNING_MATMUL_H
#define __TEST_CUDA_LEARNING_MATMUL_H
#include "element_op.h"

void callMatMulKernel(float* A,
                        float* B,
                        float* C,
                        const uint32_t m,
                        const uint32_t n,
                        const uint32_t k);


#endif //__TEST_CUDA_LEARNING_MATMUL_H