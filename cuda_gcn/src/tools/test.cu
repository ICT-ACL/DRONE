#include "cuda_runtime_api.h"
#include <cusparse_v2.h>

void print_vec(float *a, int len, char *name) {
	float* tmp;
	printf("%s:", name);
	cudaMallocHost(&tmp, sizeof(float) * len);
	cudaMemcpy(tmp, a, sizeof(float) * len, cudaMemcpyDeviceToHost);
	for (int i = 0; i < len; i++) {
		printf(" %f", tmp[i]);
	}
	printf("\n");
	cudaFreeHost(tmp);
}

void csr2csc(cusparseHandle_t handle, int *csrRowPtr, int *csrColInd, float* csrValues, int rows, int cols, int nnz) {
	size_t bufferSize;
	int *cscColPtr, *cscRowInd;
	float *cscValues;

	cudaMallocManaged(&cscColPtr, sizeof(int) * (cols + 1));
	cudaMallocManaged(&cscRowInd, sizeof(int) * nnz);
	cudaMallocManaged(&cscValues, sizeof(float) * nnz);

	cusparseCsr2cscEx2_bufferSize(handle, rows, cols, nnz, csrValues, csrRowPtr, csrColInd, cscValues, cscColPtr, cscRowInd,
								  CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &bufferSize);
	float *buffer;
	printf("csr2csc buffer size needed: %d\n", bufferSize);
	cudaMalloc(&buffer, bufferSize * sizeof(float));
	cusparseCsr2cscEx2(handle, rows, cols, nnz, csrValues, csrRowPtr, csrColInd, cscValues, cscColPtr, cscRowInd,
								  CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer);
	printf("cscColPtr:");
	for (int i = 0; i <= cols; i++) {
		printf(" %d", cscColPtr[i]);
	}
	printf("\ncscRowInd:");
	for (int i = 0; i < nnz; i++) {
		printf(" %d", cscRowInd[i]);
	}
	printf("\ncscValues:");
	for (int i = 0; i < nnz; i++) {
		printf(" %f", cscValues[i]);
	}
	printf("\n");
}

int main()
{
	cusparseHandle_t handle = NULL;
	cusparseCreate(&handle);

	cusparseSpMatDescr_t matA;
	int *csrRowOffsets;
	int csrRowOffsets_host[] = {0, 2, 4, 5};
	cudaMalloc(&csrRowOffsets, 4 * sizeof(int));
	cudaMemcpy(csrRowOffsets, csrRowOffsets_host, 4 * sizeof(int), cudaMemcpyHostToDevice);

	int nnz = 5;
	int *csrColInd;
	int csrColInd_host[] = {0, 2, 1, 3, 2};
	cudaMalloc(&csrColInd, nnz * sizeof(int));
	cudaMemcpy(csrColInd, csrColInd_host, nnz * sizeof(int), cudaMemcpyHostToDevice);

	float *csrValues;
	float csrValues_host[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
	cudaMalloc(&csrValues, nnz * sizeof(float));
	cudaMemcpy(csrValues, csrValues_host, nnz * sizeof(float), cudaMemcpyHostToDevice);


	cusparseStatus_t status = cusparseCreateCsr(&matA, 3, 4, nnz,
												csrRowOffsets, csrColInd, csrValues,
											  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
											  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
	csr2csc(handle, csrRowOffsets, csrColInd, csrValues, 3, 4, nnz);

	float *buffer;
	size_t bufferSize;





//	cusparseDnMatDescr_t matT;
//	float *tmp;
//	cudaMalloc(&tmp, 3 * 2 * sizeof(float));
//	cusparseCreateDnMat(&matT, 2, 3, 2, tmp,
//						CUDA_R_32F, CUSPARSE_ORDER_COL);
//	size_t bufferSize;
//	cusparseSparseToDense_bufferSize(handle, matA, matT,
//									 CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize);
//
//	printf("buffer size needed: %d\n", bufferSize);
//	float *buffer;
//	cudaMalloc(&buffer, bufferSize * sizeof(float));
//	cusparseSparseToDense(handle, matA, matT,
//									 CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer);
//	print_vec(tmp, 6, "MatT:");

	cusparseDnMatDescr_t matB, matC;

	float *values_host, *values;
	values_host = new float[16];
	for (int i = 0; i < 16; i++) values_host[i] = i;
	cudaMalloc(&values, 16 * sizeof(float));
	cudaMemcpy(values, values_host, 16 * sizeof(float), cudaMemcpyHostToDevice);
	cusparseCreateDnMat(&matB, 4, 4, 4, values, CUDA_R_32F, CUSPARSE_ORDER_COL);

	float *result, *result_host;
	cudaMalloc(&result, 12 * sizeof(float));
	cudaMallocHost(&result_host, 12 * sizeof(float));
//	cudaMemcpy(result, result_host, 4 * sizeof(float), cudaMemcpyHostToDevice);
	status = cusparseCreateDnMat(&matC, 3, 4, 3, result, CUDA_R_32F, CUSPARSE_ORDER_COL);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("CUSPARSE API failed at line %d with error: %s (%d)\n",
			   __LINE__, cusparseGetErrorString(status), status);
		exit(0);
	}

	float alpha = 1.0, beta = 0.0;
	status = cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
				 CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);

//	if (status != CUSPARSE_STATUS_SUCCESS) {
//		printf("CUSPARSE API failed at line %d with error: %s (%d)\n",
//			   __LINE__, cusparseGetErrorString(status), status);
//		exit(0);
//	}
	printf("buffer size needed: %d\n", bufferSize);
	cudaMalloc(&buffer, bufferSize * sizeof(float));

//	print_vec(result, 4, "result");
	status = cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
							CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer);

//	if (status != CUSPARSE_STATUS_SUCCESS) {
//		printf("CUSPARSE API failed at line %d with error: %s (%d)\n",
//			   __LINE__, cusparseGetErrorString(status), status);
//		exit(0);
//	}
//	cusparseDestroySpMat(matA);
//	cusparseDestroyDnMat(matB);
//	cusparseDestroyDnMat(matC);
//	cusparseDestroy(handle);
//
//	print_vec(csrValues, nnz, "MatA:");
//	print_vec(values, 6, "MatB:");

//	print_vec(buffer, bufferSize, "buffer:");

	cudaDeviceSynchronize();

//	printf("Result:");
//	cudaMemcpy(result_host, result, sizeof(float) * 14, cudaMemcpyDeviceToHost);
//	for (int i = 0; i < 4; i++) {
//		printf(" %f", result_host[i]);
//	}
//	printf("\n");

	print_vec(result, 12, "result");
//	cusparseDnMatGetValues(matC, (void**)&result);
//	print_vec(result, 4, "ans");
}