#ifndef __TEST_CUDA_LEARNING_MATMUL_H
#define __TEST_CUDA_LEARNING_MATMUL_H
#include "internal_include/element_op.h"

void callMatMulKernel(Tensor* A,
                        Tensor* B,
                        Tensor* C);


#endif //__TEST_CUDA_LEARNING_MATMUL_H