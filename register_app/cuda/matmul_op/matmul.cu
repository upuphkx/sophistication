#include "matmul.cuh"

template<typename T>
__global__ void MatMulKernel(T* A,
                        T* B,
                        T* C,
                        const uint32_t m,
                        const uint32_t n,
                        const uint32_t k)
{
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  float sum = 0.f;
  if (idx_x < n && idx_y < m) {
    for (int kk = 0; kk < k; ++kk) {
      sum += A[idx_y * k + kk] * B[kk * n + idx_x];
    }
    C[idx_y * n + idx_x] = sum;
  }
}

void callMatMulKernel(Tensor* A,
                        Tensor* B,
                        Tensor* C)
{
    using type = float;
    std::vector<uint32_t> shape_value_A = A->getShape()->getValue();
    std::vector<uint32_t> shape_value_B = B->getShape()->getValue();    

    type* d_A = NULL;
    type* d_B = NULL;
    type* d_C = NULL;

    cudaMalloc(reinterpret_cast<void**>(&d_A), A->getByte());
    cudaMalloc(reinterpret_cast<void**>(&d_B), B->getByte());
    cudaMalloc(reinterpret_cast<void**>(&d_C), C->getByte());        

    cudaMemcpy(d_A, reinterpret_cast<type*>(A->getBuffer()), A->getByte(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, reinterpret_cast<type*>(B->getBuffer()), B->getByte(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, reinterpret_cast<type*>(C->getBuffer()), C->getByte(), cudaMemcpyHostToDevice);

    dim3 gridDims(1, 1, 1);
    dim3 blockDims(3, 3, 1);
    MatMulKernel<<<gridDims, blockDims>>>(d_A, d_B, d_C, shape_value_A[0], shape_value_A[1], shape_value_B[0]);

    cudaMemcpy(C->getBuffer(), reinterpret_cast<void*>(d_C), C->getByte(), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}                        