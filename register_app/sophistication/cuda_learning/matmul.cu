#include "matmul.cuh"

__global__ void MatMulKernel(float* A,
                        float* B,
                        float* C,
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

void callMatMulKernel(float* A,
                        float* B,
                        float* C,
                        const uint32_t m,
                        const uint32_t n,
                        const uint32_t k)
{

    float* d_A = NULL;
    float* d_B = NULL;
    float* d_C = NULL;

    cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(float) * m * n);
    cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(float) * m * n);
    cudaMalloc(reinterpret_cast<void**>(&d_C), sizeof(float) * m * n);        

    cudaMemcpy(d_A, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeof(float) * m * n, cudaMemcpyHostToDevice);

    dim3 gridDims(1, 1, 1);
    dim3 blockDims(3, 3, 1);
    MatMulKernel<<<gridDims, blockDims>>>(d_A, d_B, d_C, m, n, k);

    cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}                        