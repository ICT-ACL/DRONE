#include "timer.h"
#include "math.h"
#include "cuda_module.cuh"
#include "cstdio"

//int local_detect_id;

//#include "seq.h"
// c(m, p) = b(n, p) * a(m, n)
CUDAMatmul::CUDAMatmul(CUDAVariable *a, CUDAVariable *b, CUDAVariable *c, int m, int n, int p, int* Mirror2Worker, string name, int workerId) :
    a(a), b(b), c(c), m(m), n(n), p(p), Mirror2Worker(Mirror2Worker), workerID(workerId) {
	ModuleName = name;
	cublasCreate(&handle);
}

void CUDAMatmul::forward(bool training) {
    timer_start(TMR_MATMUL_FW);
    timer_start(TMR_COMP);
    timer_start(TMR_TRAIN);

	float alpha, beta;
	alpha = 1.f;
	beta = 0.f;

	cublasSgemm(handle,
				CUBLAS_OP_N, // 不进行转置，则cublas其实是按照host 中转置进行计算的
				CUBLAS_OP_N, // 同上
				p, //
				m, //
				n, //
				&alpha,
				b->data, // B作为第一个OP
				p, // 由于没有使用转置, 这里填写行数
				a->data, // A作为第二个OP
				n, // 由于没有使用转置，这里填写行数
				&beta,
				c->data,
				p);

	CUDA_CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_MATMUL_FW);
    float tmp = timer_stop(TMR_COMP);
    if (time_debug) printf("matmul forward compute time:%.5f\n", tmp);

	tmp = timer_stop(TMR_TRAIN);
//	printf("Worker Id: %d, CUDAMatmul forward train time:%.5f, train time total:%.5f\n", workerID, tmp, timer_total(TMR_TRAIN));
}

__global__
void clear_mirror_grad(float* grad, int localVertexSize, int dim, int *Mirror2Worker) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	while (bid < localVertexSize) {
		if (Mirror2Worker[bid] != -1) {
			grad[bid * dim + tid] = 0.0f;
		}

		bid += gridDim.x;
	}
}

void CUDAMatmul::backward() {
	timer_start(TMR_MATMUL_BW);
    timer_start(TMR_COMP);
    timer_start(TMR_TRAIN);

	float alpha, beta;
	alpha = 1.f;
	beta = 0.f;
	if (a->requires_grad) {
		cublasSgemm(handle,
					CUBLAS_OP_T,
					CUBLAS_OP_N,
					n, //
					m, //
					p, //
					&alpha,
					b->data, // B作为第一个OP
					p, // 由于没有使用转置, 这里填写行数
					c->grad, // A作为第二个OP
					p, // 由于没有使用转置，这里填写行数
					&beta,
					a->grad,
					n);
	}
	cudaDeviceSynchronize();
	if (b->requires_grad) {
		//将c中的mirror点对应的grad置零
		clear_mirror_grad<<<max_grid_size, p>>>(c->grad, m, p, Mirror2Worker);
		cudaDeviceSynchronize();
		cublasSgemm(handle,
					CUBLAS_OP_N,
					CUBLAS_OP_T,
					p, //
					n, //
					m, //
					&alpha,
					c->grad, // B作为第一个OP
					p, // 由于没有使用转置, 这里填写行数
					a->data, // A作为第二个OP
					n, // 由于没有使用转置，这里填写行数
					&beta,
					b->grad,
					p);
	}
	cudaDeviceSynchronize();

	timer_stop(TMR_MATMUL_BW);
    float tmp = timer_stop(TMR_COMP);
    if (time_debug) printf("matmul backward compute time:%.5f\n", tmp);
	tmp = timer_stop(TMR_TRAIN);
//	printf("Worker Id: %d, CUDAMatmul backward train time:%.5f, train time total:%.5f\n", workerID, tmp, timer_total(TMR_TRAIN));
}

CUDASparseMatmul::CUDASparseMatmul(CUDAVariable *a, CUDAVariable *b, CUDAVariable *c, CUDASparseIndex *sp, int m, int n, int p) : 
    a(a), b(b), c(c), sp(sp), m(m), n(n), p(p) {}

void CUDASparseMatmul::forward(bool training) {
    timer_start(TMR_SPMATMUL_FW);

    c->zero();
    // TODO: when p larger than 1024?
    if (sp->indptr_size <= 1) return;
    dim3 block(sp->indptr_size - 1, 1, 1);
    dim3 thread_in_block(p, 1, 1);
    cuda_SparseMatmul_forward_kernel<<<block, thread_in_block>>>(a->data, b->data, c->data, sp->indptr, sp->indices, p);
    CUDA_CHECK(cudaGetLastError());

    timer_stop(TMR_SPMATMUL_FW);
}

void CUDASparseMatmul::backward() {
    timer_start(TMR_SPMATMUL_BW);

    b->zero_grad();
    // TODO: when p larger than 1024?
    if (sp->indptr_size <= 1) return;
    dim3 block(sp->indptr_size - 1, 1, 1);
    dim3 thread_in_block(p, 1, 1);
    cuda_SparseMatmul_backward_kernel<<<block, thread_in_block>>>(a->data, b->grad, c->grad, sp->indptr, sp->indices, p);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_SPMATMUL_BW);
}


CUDAGraphSum::CUDAGraphSum(CUDAVariable *in, CUDAVariable *out, Graph* graph, Comm* comm, Degree* degree, Buffer* buffer
                           , int dim, int workerId, int workerNum, string Name, int layer) :
    in(in), out(out), graph(graph), comm(comm), degree(degree), dim(dim), buffer(buffer), workerId(workerId), workerNum(workerNum), layer(layer) {
	ModuleName = Name;
	const int localVertexSize = getLocalVertexSize(graph);
	cudaMalloc(&active, sizeof(bool) * localVertexSize);

    if (batch_type == SEMI_BATCH) {
		cacheBuffer = new CacheBuffer(graph->MasterSize + graph->MirrorSize, localVertexSize, graph->Mirror2Worker, graph->MasterWorkerIndex, dim);
    }
	train_step = 0;
	cusparseCreate(&handle);

	float *csrValues = graph->edgeLen;
	cuda_compute_csr_values<<<min(max_grid_size, *graph->localVertexSize / 32 + 1), 32>>>(graph->index, graph->dst,
						degree->inDegree, degree->outDegree, *graph->localVertexSize, csrValues);
	CUDACHECK(cudaDeviceSynchronize());
	cusparseStatus_t status = cusparseCreateCsr(&matA, *graph->localVertexSize, *graph->localVertexSize, *graph->edgeSize,
												graph->index, graph->dst, csrValues,
												CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
												CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

	cudaMalloc(&cscColPtr, sizeof(int) * (*graph->localVertexSize + 1));
	cudaMalloc(&cscRowInd, sizeof(int) * *graph->edgeSize);
	cudaMalloc(&cscValues, sizeof(float) * *graph->edgeSize);

	cusparseCsr2cscEx2_bufferSize(handle, *graph->localVertexSize, *graph->localVertexSize, *graph->edgeSize, csrValues,
								  graph->index, graph->dst, cscValues, cscColPtr, cscRowInd,
								  CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &bufferSize);
	if (workerId == 0) printf("csr2csc buffer size needed: %d\n", bufferSize);
	cudaMalloc(&sparseBuffer, bufferSize * sizeof(float));
	cusparseCsr2cscEx2(handle, *graph->localVertexSize, *graph->localVertexSize, *graph->edgeSize, csrValues,
					   graph->index, graph->dst, cscValues, cscColPtr, cscRowInd,
					   CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, sparseBuffer);
	status = cusparseCreateCsr(&matAT, *graph->localVertexSize, *graph->localVertexSize, *graph->edgeSize,
							   cscColPtr, cscRowInd, cscValues, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
							   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

//	status = cusparseCreateCsc(&matAT, *graph->localVertexSize, *graph->localVertexSize, *graph->edgeSize,
//							   graph->index, graph->dst, csrValues, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//							   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

	cusparseCreateDnMat(&inDataMat, *graph->localVertexSize, dim, dim, in->data, CUDA_R_32F, CUSPARSE_ORDER_ROW);
	cusparseCreateDnMat(&outDataMat, *graph->localVertexSize, dim, dim, out->data, CUDA_R_32F, CUSPARSE_ORDER_ROW);
	cusparseCreateDnMat(&inGradMat, *graph->localVertexSize, dim, dim, in->grad, CUDA_R_32F, CUSPARSE_ORDER_ROW);
	cusparseCreateDnMat(&outGradMat, *graph->localVertexSize, dim, dim, out->grad, CUDA_R_32F, CUSPARSE_ORDER_ROW);


	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("CUSPARSE API failed at line %d with error: %s (%d)\n",
			   __LINE__, cusparseGetErrorString(status), status);
		exit(0);
	}
}

CUDAGraphSum::~CUDAGraphSum() {
	cudaFree(active);
	delete cacheBuffer;
}

// after forward, sync value (add)
void CUDAGraphSum::forward(bool training) {
    timer_start(TMR_GRAPHSUM_FW);
    timer_start(TMR_TRAIN);

    out->zero();
    const int localVertexSize = getLocalVertexSize(graph);
    float tmp;

//    timer_start(TMR_COMP);
//	cudaMemset(active, false, localVertexSize * sizeof(bool));
//    cuda_GraphSum_forward_active<<<min(102400, localVertexSize), dim>>>(in->data, out->data, graph->index, graph->dst,
//						degree->inDegree, degree->outDegree, dim, localVertexSize, active);
//
//    CUDACHECK(cudaDeviceSynchronize());
//
//    tmp = timer_stop(TMR_COMP);
//    if (time_debug) printf("graph sum forward compute time:%.5f\n", tmp);

	CUDACHECK(cudaDeviceSynchronize());
    timer_start(TMR_COMP);
//    cuda_GraphSum_forward<<<min(102400, localVertexSize), dim>>>(in->data, out->data, graph->index, graph->dst,
//                                                                 degree->inDegree, degree->outDegree, dim, localVertexSize);

	float alpha = 1.0, beta = 0.0;
	if (train_step == 0) {
		size_t bufferSizeTmp;
		cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matAT, inDataMat, &beta, outDataMat,
								CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSizeTmp);
		if (bufferSize < bufferSizeTmp) {
			bufferSize = bufferSizeTmp;
			printf("buffer size needed: %d\n", bufferSize);
			cudaFree(sparseBuffer);
			cudaMalloc(&sparseBuffer, bufferSize * sizeof(float));
		}
	}
	cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matAT, inDataMat, &beta, outDataMat,
						  CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, sparseBuffer);

    CUDACHECK(cudaDeviceSynchronize());
    tmp = timer_stop(TMR_COMP);
    if (time_debug) printf("graph sum forward compute time ori:%.5f\n", tmp);

//    timer_start(TMR_COMM_);
//    GraphSum_sync_value_nccl_active(out, buffer, comm, graph, dim, workerId, workerNum, active);
//    CUDACHECK(cudaDeviceSynchronize());
//    CUDA_CHECK(cudaGetLastError());
//    tmp = timer_stop(TMR_COMM_);
//    if (time_debug) printf("graph sum forward communicate time:%.5f\n", tmp);

//    int detect_global_id = 2019;
//    local_detect_id = graph->Global2Local[detect_global_id];
//    float *invalue, *outvalue, *outLocal;
//    if (local_detect_id != -1) {
//        cudaMallocHost(&outLocal, sizeof(float) * dim);
//        cudaMemcpy(outLocal, out->data + local_detect_id * dim, sizeof(float) * dim, cudaMemcpyDeviceToHost);
//    }

	timer_start(TMR_COMM_);
	if (train_step == 0 || train_step % 100 == 5 || batch_type != SEMI_BATCH) {
		GraphSum_sync_value_nccl(out, buffer, comm, graph, cacheBuffer, dim, workerId, workerNum);
	} else {
		GraphSum_sync_value_nccl_cache(out, buffer, comm, graph, cacheBuffer, dim, workerId, workerNum, active, layer);
	}

	CUDACHECK(cudaDeviceSynchronize());

//    if (local_detect_id != -1) {
//        if (graph->MasterWorkerIndex[local_detect_id + 1] - graph->MasterWorkerIndex[local_detect_id] > 0) {
//            printf("Master vertex:%d in %d info:", detect_global_id, workerId);
//            for (int ind = graph->MasterWorkerIndex[local_detect_id]; ind < graph->MasterWorkerIndex[local_detect_id + 1]; ind++) {
//                printf(" %d", graph->Master2Workers[ind]);
//            }
//            printf("\n");
//        }
//        if (graph->Mirror2Worker[local_detect_id] != -1) {
//            printf("Mirror vertex:%d in %d info: %d\n", detect_global_id, workerId, graph->Mirror2Worker[local_detect_id]);
//        }
//        cudaMallocHost(&invalue, sizeof(float)  * dim);
//        cudaMemcpy(invalue, in->data + local_detect_id * dim, sizeof(float) * dim, cudaMemcpyDeviceToHost);
//        cudaMallocHost(&outvalue, sizeof(float) * dim);
//        cudaMemcpy(outvalue, out->data + local_detect_id * dim, sizeof(float) * dim, cudaMemcpyDeviceToHost);
//
//        for (int i = 0; i < 5; i++) {
//            printf("Worker %d, dim:%d, local_id:%d, invalue:%f, outLocal:%f, outsync:%f\n", workerId, i, local_detect_id, invalue[i], outLocal[i], outvalue[i]);
//        }
//    }
//    exit(0);


	tmp = timer_stop(TMR_COMM_);
//	if (time_debug) printf("Worker Id: %d, graph sum forward communicate time ori:%.5f\n", workerId, tmp);

//    if (time_debug) printf("-------\n");
    timer_stop(TMR_GRAPHSUM_FW);
	tmp = timer_stop(TMR_TRAIN);
//	printf("Worker Id: %d, graph sum forward train time:%.5f, train time total:%.5f\n", workerId, tmp, timer_total(TMR_TRAIN));
//    printf("--------------------\n");
}

// after backward , sync grad (add reverse)
void CUDAGraphSum::backward() {
    timer_start(TMR_GRAPHSUM_BW);
    timer_start(TMR_COMP);
    timer_start(TMR_TRAIN);

    in->zero_grad();
    const int localVertexSize = getLocalVertexSize(graph);
//	cudaMemset(active, false, localVertexSize * sizeof(bool));
//    cuda_GraphSum_backward_active<<<min(102400, localVertexSize), dim>>>(in->grad, out->grad, graph->index, graph->dst,
//             degree->inDegree, degree->outDegree, dim, localVertexSize, active);
//    cuda_GraphSum_backward<<<min(102400, localVertexSize), dim>>>(in->grad, out->grad, graph->index, graph->dst,
//                                                                         degree->inDegree, degree->outDegree, dim, localVertexSize);

	float alpha = 1.0, beta = 0.0;
	if (train_step == 0) {
		size_t bufferSizeTmp;
		cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, outGradMat, &beta, inGradMat,
								CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSizeTmp);
		if (bufferSize < bufferSizeTmp) {
			bufferSize = bufferSizeTmp;
			cudaFree(sparseBuffer);
			printf("buffer size needed: %d\n", bufferSize);
			cudaMalloc(&sparseBuffer, bufferSize * sizeof(float));
		}
	}
	cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, outGradMat, &beta, inGradMat,
							CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, sparseBuffer);

    CUDA_CHECK(cudaGetLastError());
	CUDACHECK(cudaDeviceSynchronize());
    float tmp = timer_stop(TMR_COMP);
    if (time_debug) printf("graph sum backward compute time:%.5f\n", tmp);

//    float *ingrad, *outgrad, *invalue, *outvalue;
//    int test_len = 10;
//    cudaMallocHost(&ingrad, sizeof(float) * test_len * dim);
//    cudaMemcpy(ingrad, in->grad, sizeof(float) * test_len * dim, cudaMemcpyDeviceToHost);
//    cudaMallocHost(&outgrad, sizeof(float) * test_len * dim);
//    cudaMemcpy(outgrad, out->grad, sizeof(float) * test_len * dim, cudaMemcpyDeviceToHost);
//    cudaMallocHost(&invalue, sizeof(float) * test_len * dim);
//    cudaMemcpy(invalue, in->data, sizeof(float) * test_len * dim, cudaMemcpyDeviceToHost);
//    cudaMallocHost(&outvalue, sizeof(float) * test_len * dim);
//    cudaMemcpy(outvalue, out->data, sizeof(float) * test_len * dim, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < test_len; i++) {
//        printf("layer:%d, id:%d, in->grad:%e, out->grad:%e, in->value:%e, out->value:%e\n", layer, i, ingrad[i], outgrad[i], invalue[i], outvalue[i]);
//    }

//	float *ingrad, *outgrad, *inLocal;
//    if (local_detect_id != -1) {
//        cudaMallocHost(&inLocal, sizeof(float) * dim);
//        cudaMemcpy(inLocal, in->grad + local_detect_id * dim, sizeof(float) * dim, cudaMemcpyDeviceToHost);
//    }

    timer_start(TMR_COMM_);
    if (train_step % 10 == 0 || batch_type != SEMI_BATCH) {
        GraphSum_sync_grad_nccl(in, buffer, comm, graph, cacheBuffer, dim, workerId, workerNum);
    } else {
        GraphSum_sync_grad_nccl_cache(in, buffer, comm, graph, cacheBuffer, dim, workerId, workerNum, active, layer);
    }
    CUDACHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

//    if (local_detect_id != -1) {
//        cudaMallocHost(&ingrad, sizeof(float)  * dim);
//        cudaMemcpy(ingrad, in->grad + local_detect_id * dim, sizeof(float) * dim, cudaMemcpyDeviceToHost);
//        cudaMallocHost(&outgrad, sizeof(float) * dim);
//        cudaMemcpy(outgrad, out->grad + local_detect_id * dim, sizeof(float) * dim, cudaMemcpyDeviceToHost);
//
//        for (int i = 10; i < 15; i++) {
//            printf("Worker %d, dim:%d, local_id:%d, insync:%e, inLocal:%e, outgrad:%e\n", workerId, i, local_detect_id, ingrad[i], inLocal[i], outgrad[i]);
//        }
//    }

    tmp = timer_stop(TMR_COMM_);
    if (time_debug) printf("graph sum backward communicate time:%.5f\n", tmp);
    timer_stop(TMR_GRAPHSUM_BW);
	tmp = timer_stop(TMR_TRAIN);
//	printf("Worker Id: %d, graph sum backward train time:%.5f, train time total:%.5f\n", workerId, tmp, timer_total(TMR_TRAIN));
    train_step++;
}

CUDACrossEntropyLoss::CUDACrossEntropyLoss(CUDAVariable *logits, int *truth, float *loss, int num_classes, string name) :
    logits(logits), truth(truth), loss(loss), num_classes(num_classes) {
    int logitsPerClass = logits->size / num_classes;
    CUDA_CHECK(cudaMalloc((void**) &d_loss, logitsPerClass * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &d_count, logitsPerClass * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged((void**) &logits_max, logits->size / num_classes * sizeof(float)));
	ModuleName = name;
}

CUDACrossEntropyLoss::~CUDACrossEntropyLoss() {
    CUDA_CHECK(cudaFree(d_loss));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(logits_max));
}

void CUDACrossEntropyLoss::forward(bool training) {
    timer_start(TMR_LOSS_FW);

    if (training) logits->zero_grad();
    
    int logitsPerClass = logits->size / num_classes;

    CUDA_CHECK(cudaMemset(d_loss, 0, logitsPerClass * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_count, 0, logitsPerClass * sizeof(int)));

//    if (local_detect_id != -1) {
//        int *truth_host;
//        cudaMallocHost(&truth_host, sizeof(int));
//        cudaMemcpy(truth_host, truth + local_detect_id, sizeof(int), cudaMemcpyDeviceToHost);
//        printf("truth[%d]:%d\n", local_detect_id, *truth_host);
//    }

    int stride;
    for (int k = 0; k < 32; k++) {
        if ((num_classes - 1) >> k == 1) {
            stride = 1 << k;
            break;
        }
    }
    assert(stride <= 32);
    cal_max<<<min(max_grid_size, logitsPerClass), stride, sizeof(float) * stride>>>(logits->data, logits_max, logitsPerClass, num_classes, truth, stride);

	cudaDeviceSynchronize();
//	if (local_detect_id != -1) {
//		int *truth_host;
//		cudaMallocHost(&truth_host, sizeof(int));
//		cudaMemcpy(truth_host, truth + local_detect_id, sizeof(int), cudaMemcpyDeviceToHost);
//		printf("logits_max: %e, truth:%d\n", logits_max[local_detect_id], *truth_host);
//	}

    cuda_CrossEntropy_forward_A_kernel_<<<min(max_grid_size, logitsPerClass), num_classes>>>
    (logits->data, logits_max, logits->grad, training, num_classes, truth, d_count, d_loss, logitsPerClass);

    CUDA_CHECK(cudaGetLastError());

//    dim3 block(logitsPerClass, 1, 1);
//    dim3 thread_in_block(1, 1, 1);
//    cuda_CrossEntropy_forward_A_kernel<<<block, thread_in_block>>>(logits->data, logits->grad, training, num_classes, truth, d_count, d_loss, logits->size);

    thrust::device_ptr<int> count_ptr = thrust::device_pointer_cast(d_count);
    int count = thrust::reduce(count_ptr, count_ptr + logitsPerClass, (int)0, thrust::plus<int>());
	int count_grad = 1;
//	int count_grad = 196615;
    thrust::device_ptr<float> loss_ptr = thrust::device_pointer_cast(d_loss);
    *loss = thrust::reduce(loss_ptr, loss_ptr + logitsPerClass, (float)0.0, thrust::plus<float>());
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

//    printf("CUDACrossEntropyLoss: count:%d\n", count);

    *loss /= count;
    dim3 block2(logits->size, 1, 1);
    dim3 thread_in_block2(1, 1, 1);
    if (training) {
        cuda_CrossEntropy_forward_B_kernel<<<block2, thread_in_block2>>>(logits->grad, logits->size, count_grad);
        CUDA_CHECK(cudaGetLastError());
        // CUDA_CHECK(cudaDeviceSynchronize());
    }

//	float *value, *grad;
//	if (local_detect_id != -1) {
//		cudaMallocHost(&value, sizeof(float) * num_classes);
//		cudaMemcpy(value, logits->data + local_detect_id * num_classes, sizeof(float) * num_classes, cudaMemcpyDeviceToHost);
//		cudaMallocHost(&grad, sizeof(float) * num_classes);
//		cudaMemcpy(grad, logits->grad + local_detect_id * num_classes, sizeof(float) * num_classes, cudaMemcpyDeviceToHost);
//
//		for (int i = 10; i < 15; i++) {
//			printf("CrossEntropy, local_id:%d, value:%e, grad:%e\n", i, local_detect_id, value[i], grad[i]);
//		}
//	}

    timer_stop(TMR_LOSS_FW);
}

void CUDACrossEntropyLoss::backward() {
}

CUDAReLU::CUDAReLU(CUDAVariable *in, string name) :
    in(in) {
    CUDA_CHECK(cudaMalloc((void**) &mask, in->size * sizeof(bool)));
	ModuleName = name;
}

CUDAReLU::~CUDAReLU() {
    CUDA_CHECK(cudaFree(mask));
}

void CUDAReLU::forward(bool training) {
    timer_start(TMR_RELU_FW);

    dim3 block((in->size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    CUDA_CHECK(cudaGetLastError());
    cuda_ReLU_forward_kernel<<<block, thread_in_block>>>(in->data, mask, in->size, training);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_RELU_FW);
}

void CUDAReLU::backward() {
    timer_start(TMR_RELU_BW);

    dim3 block((in->size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize())
    cuda_ReLU_backward_kernel<<<block, thread_in_block>>>(in->grad, mask, in->size);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_RELU_BW);
}

CUDADropout::CUDADropout(CUDAVariable *in, float p, string name) :
    in(in), p(p) {
	ModuleName = name;
    if (in->requires_grad) {
        CUDA_CHECK(cudaMalloc((void**) &mask, in->size * sizeof(int)));
    }
    else
        mask = nullptr;
}

CUDADropout::~CUDADropout() {
    if (mask != nullptr) CUDA_CHECK(cudaFree(mask));
}

void CUDADropout::forward(bool training) {
    if (!training) return;
    timer_start(TMR_DROPOUT_FW);
//    timer_start(TMR_COMP);

    float scale = 1 / (1 - p);
    dim3 block((in->size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    cuda_Dropout_forward_kernel<<<block, thread_in_block>>>(in->data, mask, devStates, in->size, p, scale, (mask != nullptr));
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
//    timer_stop(TMR_COMP);
    timer_stop(TMR_DROPOUT_FW);
}

void CUDADropout::backward() {
    if (mask == nullptr) return;
    timer_start(TMR_DROPOUT_BW);
    float scale = 1 / (1 - p);
    dim3 block((in->size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    cuda_Dropout_backward_kernel<<<block, thread_in_block>>>(in->grad, mask, in->size, scale);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
    timer_stop(TMR_DROPOUT_BW);
}

CUDAAdamVariable::CUDAAdamVariable(CUDAVariable *var, bool decay) :
    data(var->data), grad(var->grad), size(var->size), decay(decay) {
    CUDA_CHECK(cudaMalloc((void**) &m, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &v, size * sizeof(float)));
}

CUDAAdamVariable::~CUDAAdamVariable() {
    CUDA_CHECK(cudaFree(m));
    CUDA_CHECK(cudaFree(v));
}

CUDAAdam::CUDAAdam(vector<pair<CUDAVariable*, bool>> vars, AdamParams params) :
    step_count(0), params(params){
    for (auto v : vars) {
        CUDAAdamVariable *adam_var = new CUDAAdamVariable(v.first, v.second);
        this->vars.push_back(adam_var);
    }
}

CUDAAdam::~CUDAAdam() {
    for (auto &var : vars)
        delete var;
}

void CUDAAdam::step() {
    step_count++;
    float step_size = params.lr * sqrtf(1 - powf(params.beta2, step_count)) / (1 - powf(params.beta1, step_count));
    for (auto &var : vars) {
        dim3 block((var->size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
        dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
        cuda_Adam_step_kernel<<<block, thread_in_block>>>(var->grad, var->data, var->m, var->v, var->decay, params.weight_decay, params.beta1, params.beta2, params.eps, step_size, var->size);
        CUDA_CHECK(cudaGetLastError());
        // CUDA_CHECK(cudaDeviceSynchronize());
    }
}
