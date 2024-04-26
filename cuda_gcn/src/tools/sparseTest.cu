#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

// m * n (row major) -> n * m (column major) transpose
void Transpose(cublasHandle_t handle, int m, int n, float *d_A, float *d_A_T)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgeam(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n,
                &alpha,
                d_A, n,
                &beta,
                d_A, m,
                d_A_T, m);
}

void printPlainMatrix(const float* matrix, const int H, const int W)
{
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            std::cout << std::fixed << std::setw(8) << std::setprecision(4) << matrix[i * W + j];
        }
        std::cout << std::endl;
    }
}

//void csr2bsr() {
//    // Suppose that A is m x n sparse matrix represented by CSR format,
//    // hx is a host vector of size n, and hy is also a host vector of size m.
//    // m and n are not multiple of blockDim.
//    // step 1: transform CSR to BSR with column-major order
//    int base, nnz;
//    int nnzb;
//    cusparseDirection_t dirA = CUSPARSE_DIRECTION_COLUMN;
//    int mb = (m + blockDim-1)/blockDim;
//    int nb = (n + blockDim-1)/blockDim;
//    cudaMalloc((void**)&bsrRowPtrC, sizeof(int) *(mb+1));
//    cusparseXcsr2bsrNnz(handle, dirA, m, n,
//                        descrA, csrRowPtrA, csrColIndA, blockDim,
//                        descrC, bsrRowPtrC, &nnzb);
//    cudaMalloc((void**)&bsrColIndC, sizeof(int)*nnzb);
//    cudaMalloc((void**)&bsrValC, sizeof(float)*(blockDim*blockDim)*nnzb);
//    cusparseScsr2bsr(handle, dirA, m, n,
//                     descrA, csrValA, csrRowPtrA, csrColIndA, blockDim,
//                     descrC, bsrValC, bsrRowPtrC, bsrColIndC);
//}

// Example usage
int main()
{
    const int n = 2;
    const int m = 3;
    const int k = 4;

    float *b, *c, *bt;
    float alpha, beta;
    alpha = 1.0;
    beta = 0.0;

    // Create a Cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    cusparseHandle_t sparse_handle;
    cusparseCreate(&sparse_handle);

    cudaMallocManaged(&b, sizeof(float) * m * k);
    cudaMallocManaged(&c, sizeof(float) * m * k);
    cudaMallocManaged(&bt, sizeof(float) * m * k);

    for (int i = 0; i < m; i++) for (int j = 0; j < k; j++) {
        b[i * k + j] = i * k + j;
    }
    printf("B ori:\n");
    printPlainMatrix(b, m, k);

    Transpose(handle, m, k, b, bt);
    cudaDeviceSynchronize();
    printf("B transpose:\n");
    printPlainMatrix(bt, k, m);

    int nnz = 3;
    float *csr_val, *csrRowPtr, *csrColInd;
    cudaMallocManaged(&csr_val, sizeof(float) * nnz);
    csr_val[0] = 1; csr_val[1] = 2; csr_val[2] = 3;
    cudaMallocManaged(&csrRowPtr, sizeof(float) * (n + 1));
    csrRowPtr[0] = 0;
    csrRowPtr[1] = 2;
    csrRowPtr[2] = 3;
    cudaMallocManaged(&csrColInd, sizeof(float) * nnz);
    csrColInd[0] = 0; csrColInd[1] = 2; csrColInd[2] = 1;

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    cusparseScsrmm_batched(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, k, m, nnz, &alpha, descr,csr_val,csrRowPtr,csrColInd, bt, m, &beta,c, n);

    printf("C:\n");
    printPlainMatrix(c, n, k);

    // Destroy the Cublas handle and free memory on the device
    cublasDestroy(handle);
    cusparseDestroy(sparse_handle);

    return 0;
}