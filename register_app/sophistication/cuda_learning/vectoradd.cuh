#ifndef __VECTOR_ADD_CUH
#define __VECTOR_ADD_CUH
#include "element_op.h"
void CallVectorAddKernelFunction(std::vector<ElementType> input_1, 
                                std::vector<ElementType> input_2, 
                                ElementType* output,
                                uint32_t ElementNum);                             
#endif //__VECTOR_ADD_CUH