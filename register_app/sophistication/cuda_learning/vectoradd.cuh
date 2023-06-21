#ifndef __VECTOR_ADD_CUH
#define __VECTOR_ADD_CUH
#include "element_op.h"

void CallVectorAddKernelFunction(Tensor* input_1, 
                                Tensor*  input_2, 
                                Tensor*  output);           
#endif //__VECTOR_ADD_CUH