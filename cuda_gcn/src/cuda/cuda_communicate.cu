#include "timer.h"
#include "bits/stdc++.h"
#include "../common/utils.h"
#include "cuda_cache.cuh"
#include "cuda_communicate.cuh"

__global__
void calSendRecvLen(int *mirror2masterLen, int *master2mirrorLen, int *Mirror2Worker,
					int *MasterWorkerIndex, int *Master2Workers, int localVertexSize) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	while (id < localVertexSize) {
		if (Mirror2Worker[id] != -1) atomicAdd(mirror2masterLen + Mirror2Worker[id], 1);

		for (int i = MasterWorkerIndex[id]; i < MasterWorkerIndex[id + 1]; i++) {
			int worker = Master2Workers[i];
			int *address = master2mirrorLen + worker;
			atomicAdd(address, 1);
		}
		id += stride;
	}

}

void generateLen(Buffer* buffer, int bufferSize, Graph* graph, int workerId, int workerNum) {
//    printf("workerNum: %d\n", workerNum);
	for (int i = 0; i < workerNum; i++) {
		buffer->master2mirrorLen[i] = 0;
		buffer->mirror2masterLen[i] = 0;
	}
	int localVertexSize = getLocalVertexSize(graph);
	calSendRecvLen<<<graph->gridSize, graph->blockSize>>>(buffer->mirror2masterLen, buffer->master2mirrorLen, graph->Mirror2Worker, graph->MasterWorkerIndex, graph->Master2Workers, localVertexSize);
    cudaDeviceSynchronize();
	buffer->mirror2masterStart[0] = 0;
	buffer->master2mirrorStart[0] = 0;
	for (int i = 1; i <= workerNum; i++) {
		buffer->mirror2masterStart[i] = buffer->mirror2masterStart[i - 1] + buffer->mirror2masterLen[i - 1];
		buffer->master2mirrorStart[i] = buffer->master2mirrorStart[i - 1] + buffer->master2mirrorLen[i - 1];
	}
}

void buffer_init(Buffer* buffer, Graph* graph, int workerId, int workerNum) {
	int bufferSize = max(graph->MirrorSize, graph->MirrorWorkerSize);
    printf("bufferSize:%d, maxDim:%d\n", bufferSize, buffer->maxDim);

    if (compress) {
        cudaMalloc(&buffer->sendValue_Int, sizeof(uint8_t) * bufferSize * buffer->maxDim);
        cudaMalloc(&buffer->recvValue_Int, sizeof(uint8_t) * bufferSize * buffer->maxDim);
		cudaMalloc(&buffer->maxminSend, sizeof(float) * bufferSize * 2);
		cudaMalloc(&buffer->maxminRecv, sizeof(float) * bufferSize * 2);
        cudaMallocManaged(&buffer->value, sizeof(float) * bufferSize * buffer->maxDim);
    }

	cudaMalloc(&buffer->sendValue, sizeof(float) * bufferSize * buffer->maxDim);
	cudaMalloc(&buffer->recvValue, sizeof(float) * bufferSize * buffer->maxDim);
	cudaMalloc(&buffer->sendID, sizeof(int) * bufferSize);
	cudaMalloc(&buffer->recvID, sizeof(int) * bufferSize);

	cudaMallocManaged(&buffer->mirror2masterLen, sizeof(int) * workerNum);
	cudaMallocManaged(&buffer->master2mirrorLen, sizeof(int) * workerNum);
    cudaMallocManaged(&buffer->recvLen, sizeof(int) * workerNum);

	cudaMallocManaged(&buffer->mirror2masterIdx, sizeof(int) * (workerNum + 1));
	cudaMallocManaged(&buffer->master2mirrorIdx, sizeof(int) * (workerNum + 1));

	cudaMallocManaged(&buffer->mirror2masterStart, sizeof(int) * (workerNum + 1));
	cudaMallocManaged(&buffer->master2mirrorStart, sizeof(int) * (workerNum + 1));

    CUDACHECK(cudaDeviceSynchronize());

	generateLen(buffer, bufferSize, graph, workerId, workerNum);
    CUDACHECK(cudaDeviceSynchronize());
}

void delete_buffer(Buffer* buffer) {
	cudaFree(buffer->sendValue);
	cudaFree(buffer->recvValue);
	cudaFree(buffer->sendID);
	cudaFree(buffer->recvID);
    if (compress) {
        cudaFree(buffer->sendValue_Int);
        cudaFree(buffer->recvValue_Int);
		cudaFree(buffer->maxminSend);
		cudaFree(buffer->maxminRecv);
        cudaFree(buffer->value);
    }
	cudaFree(buffer->mirror2masterLen);
	cudaFree(buffer->master2mirrorLen);
    cudaFree(buffer->recvLen);

	cudaFree(buffer->mirror2masterIdx);
	cudaFree(buffer->master2mirrorIdx);

	cudaFree(buffer->mirror2masterStart);
	cudaFree(buffer->master2mirrorStart);
}

void weight_broadcast_value_nccl(vector<CUDAVariable>* variables, Comm* comm) {
    for (int i = 0; i < variables->size(); i++) {
        if (!(*variables)[i].is_weight) continue;
		NCCLCHECK(ncclBcast((*variables)[i].data, (*variables)[i].size, ncclFloat, 0, comm->comm, comm->s));
		CUDA_CHECK(cudaStreamSynchronize(comm->s));
    }
}

void weight_reduce_grad_nccl(vector<CUDAVariable>* variables, Comm* comm) {
	for (int i = 0; i < (*variables).size(); i++) {
		if (!(*variables)[i].is_weight) continue;

		NCCLCHECK(ncclReduce((*variables)[i].grad, (*variables)[i].grad, (*variables)[i].size, ncclFloat, ncclSum, 0, comm->comm, comm->s));
//		NCCLCHECK(ncclBcast((*variables)[i].grad, (*variables)[i].size, ncclFloat, 0, comm->comm, comm->s));
		CUDA_CHECK(cudaStreamSynchronize(comm->s));
	}
}

void buffer_initIdx(Buffer* buffer, int workerNum) {
    buffer->mirror2masterIdx[0] = 0;
    buffer->master2mirrorIdx[0] = 0;
    for (int i = 1; i <= workerNum; i++) {
        buffer->mirror2masterIdx[i] = buffer->mirror2masterIdx[i - 1] + buffer->mirror2masterLen[i - 1];
        buffer->master2mirrorIdx[i] = buffer->master2mirrorIdx[i - 1] + buffer->master2mirrorLen[i - 1];
    }
}

__global__
void set_sync_value_mirror2master(int* sendID, float* sendValue, float* value, int feature_size,
                                  int* mirror2masterIdx, int* Mirror2Worker, int* Local2Global, int localVertexSize) {
    int stride = gridDim.x;
    int id = blockIdx.x;
    __shared__ int idx[2];
    while (id < localVertexSize) {
        if (Mirror2Worker[id] == -1) {
            id += stride;
            continue;
        }

        if (threadIdx.x == 0) {
            idx[0] = atomicAdd(mirror2masterIdx + Mirror2Worker[id], 1);
            sendID[idx[0]] = Local2Global[id];
            idx[0] *= feature_size;
            idx[1] = id * feature_size;
        }
        __syncthreads();
        sendValue[idx[0] + threadIdx.x] = value[idx[1] + threadIdx.x];

//		if (Local2Global[id] == 2019 && threadIdx.x >= 10 && threadIdx.x < 15) {
//			printf("mirror2master send %d to %d: idx:%d, value:%e\n", Local2Global[id], Mirror2Worker[id], threadIdx.x, sendValue[idx[0] + threadIdx.x]);
//		}

        id += stride;
    }
}

__global__
void set_sync_value_mirror2master_debug(int* sendID, float* sendValue, float* value, int feature_size,
                                  int* mirror2masterIdx, int* Mirror2Worker, int* Local2Global, int localVertexSize, bool* active) {
	int stride = gridDim.x;
	int id = blockIdx.x;
	__shared__ int idx[2];
	while (id < localVertexSize) {
		if (Mirror2Worker[id] == -1 || !active[id]) {
			id += stride;
			continue;
		}

		if (threadIdx.x == 0) {
			idx[0] = atomicAdd(mirror2masterIdx + Mirror2Worker[id], 1);
			sendID[idx[0]] = Local2Global[id];
			idx[0] *= feature_size;
			idx[1] = id * feature_size;
		}
		__syncthreads();
		sendValue[idx[0] + threadIdx.x] = value[idx[1] + threadIdx.x];

		id += stride;
	}
}

//int *MasterWorkerIndex; //  master local id -> mirror worker location list in CSR format
//int *Master2Workers;

__global__
void set_sync_value_master2mirror(int* sendID, float* sendValue, float* value, int feature_size, int* master2mirrorIdx,
								  int* Master2Workers, int *MasterWorkerIndex, int* Local2Global, int localVertexSize, int workerNum) {
	extern __shared__ int idx[];
	int id = blockIdx.x;
	int stride = gridDim.x;

	while (id < localVertexSize) {
		const int searchLen = MasterWorkerIndex[id + 1] - MasterWorkerIndex[id];
		int mirrorId = threadIdx.x;
		while (mirrorId < searchLen) {
			int worker = Master2Workers[mirrorId + MasterWorkerIndex[id]];
			idx[mirrorId] = atomicAdd(master2mirrorIdx + worker, 1);
			sendID[idx[mirrorId]] = Local2Global[id];
			mirrorId += blockDim.x;
		}
		if (threadIdx.x == 0) idx[workerNum - 1] = searchLen;
		__syncthreads();
		for (int i = 0; i < idx[workerNum - 1]; i++) {
			sendValue[idx[i] * feature_size + threadIdx.x] = value[id * feature_size + threadIdx.x];
		}

		id += stride;
	}
}

__global__
void set_sync_value_master2mirror_debug(int* sendID, float* sendValue, float* value, int feature_size, int* master2mirrorIdx,
                                  int* Master2Workers, int *MasterWorkerIndex, int* Local2Global, int localVertexSize, int workerNum, bool* active) {
	extern __shared__ int idx[];
	int id = blockIdx.x;
    int stride = gridDim.x;

    while (id < localVertexSize) {
		const int searchLen = MasterWorkerIndex[id + 1] - MasterWorkerIndex[id];
        if (searchLen == 0 || !active[id]) {
            id += stride;
            continue;
        }
		int mirrorId = threadIdx.x;
		while (mirrorId < searchLen) {
			int worker = Master2Workers[mirrorId + MasterWorkerIndex[id]];
			idx[mirrorId] = atomicAdd(master2mirrorIdx + worker, 1);
			sendID[idx[mirrorId]] = Local2Global[id];
			mirrorId += blockDim.x;
		}
		if (threadIdx.x == 0) idx[workerNum - 1] = searchLen;
		__syncthreads();
		for (int i = 0; i < idx[workerNum - 1]; i++) {
			sendValue[idx[i] * feature_size + threadIdx.x] = value[id * feature_size + threadIdx.x];
		}

        id += stride;
    }
}

__global__
void apply_sum(float* data, int* recvID, float* recvValue, int* Global2Local, int recvLen, int feature_size) {
	int id = blockIdx.x;
	int stride = gridDim.x;

	while (id < recvLen) {
		int localID = Global2Local[recvID[id]];
		atomicAdd(data + localID * feature_size + threadIdx.x, recvValue[id * feature_size + threadIdx.x]);
		id += stride;
	}
}

//__global__
//void apply_sum_active(float* data, int* recvID, float* recvValue, int* Global2Local, int recvLen, int feature_size, bool* active) {
//	int id = blockIdx.x;
//	int stride = gridDim.x;
//
//	while (id < recvLen) {
//		int localID = Global2Local[recvID[id]];
//		atomicAdd(data + localID * feature_size + threadIdx.x, recvValue[id * feature_size + threadIdx.x]);
//        if (threadIdx.x == 0) active[localID] = true;
//		id += stride;
//	}
//}

__global__
void apply_assign(float* data, int* recvID, float* recvValue, int* Global2Local, int recvLen, int feature_size) {
	int id = blockIdx.x;
	int stride = gridDim.x;

	while (id < recvLen) {
		int localID = Global2Local[recvID[id]];
		data[localID * feature_size + threadIdx.x] = recvValue[id * feature_size + threadIdx.x];
		id += stride;
	}
}

void sync_recvLen(const int *sendStart, const int *sendEnd, int *recvLen, Comm* comm, int workerId, int workerNum) {
    ncclGroupStart();
    int *sendLen;
    cudaMallocHost(&sendLen, sizeof(int) * workerNum);
    for (int i = 0; i < workerNum; i++) sendLen[i] = sendEnd[i] - sendStart[i];
    for (int i = 0; i < workerNum; i++) {
        if (i == workerId) {
            recvLen[i] = 0;
            continue;
        }
        ncclSend(sendLen + i, 1, ncclInt, i, comm->comm, comm->s);
        ncclRecv(recvLen + i, 1, ncclInt, i, comm->comm, comm->s);
//		printf("send to %d, sendIdx:%d, sendLen%d\n", i, sendIdx[i], sendLen[i]);
    }
    ncclGroupEnd();
    cudaStreamSynchronize(comm->s);
//    for (int i = 0; i < workerNum; i++) printf("recv from %d,  recvLen:%d\n", i, recvLen[i]);
    cudaFreeHost(sendLen);
}

int sendID_Value(Buffer* buffer, int* sendStart, int* sendEnd, int *recvLen, Comm* comm, int feature_size, int workerId, int workerNum) {
//	timer_start(TMR_TMP);
    sync_recvLen(sendStart, sendEnd, recvLen, comm, workerId, workerNum);

    int *recvIndex = new int[workerNum + 1];
    recvIndex[0] = 0;
    for (int i = 1; i <= workerNum; i++) {
        recvIndex[i] = recvIndex[i - 1] + recvLen[i - 1];
    }
    ncclGroupStart();
    for (int i = 0; i < workerNum; i++) {
        if (i == workerId) continue;
        ncclSend(buffer->sendID + sendStart[i], sendEnd[i] - sendStart[i], ncclInt, i, comm->comm, comm->s);
        ncclRecv(buffer->recvID + recvIndex[i], recvLen[i], ncclInt, i, comm->comm, comm->s);
//		printf("send to %d, sendStart:%d, sendEnd:%d\n", i, sendStart[i], sendEnd[i]);
//		printf("recv from %d, recvIndex:%d, recvLen:%d\n", i, recvIndex[i], recvLen[i]);
    }
    ncclGroupEnd();
    cudaStreamSynchronize(comm->s);
//	printf("send ID time: %.6f, len:%d\n", timer_stop(TMR_TMP), sendEnd[workerNum]);

//	timer_start(TMR_TMP);
	if (compress) {
		compressFloat2Uint8(buffer->sendValue, buffer->sendValue_Int, sendEnd[workerNum], feature_size, buffer->maxminSend);
//		printf("compress time: %.6f, size:%d * %d\n", timer_stop(TMR_TMP), sendEnd[workerNum], feature_size);

		ncclGroupStart();
//		timer_start(TMR_TMP);
		int sendLen = 0;
		for (int i = 0; i < workerNum; i++) {
			if (i == workerId) continue;
			ncclSend(buffer->sendValue_Int + sendStart[i] * feature_size, (sendEnd[i] - sendStart[i]) * feature_size,
					 ncclUint8, i, comm->comm, comm->s);
			ncclRecv(buffer->recvValue_Int + recvIndex[i] * feature_size, recvLen[i] * feature_size,
					 ncclUint8, i, comm->comm, comm->s);
			sendLen += sendEnd[i] - sendStart[i];
		}
		ncclGroupEnd();
		cudaStreamSynchronize(comm->s);
//		printf("send compressed value time: %.6f, len:%d\n", timer_stop(TMR_TMP), sendLen);

		ncclGroupStart();
//		timer_start(TMR_TMP);
		for (int i = 0; i < workerNum; i++) {
			if (i == workerId) continue;
			ncclSend(buffer->maxminSend + sendStart[i] * 2, (sendEnd[i] - sendStart[i]) * 2,
					 ncclFloat, i, comm->comm, comm->s);
			ncclRecv(buffer->maxminRecv + recvIndex[i] * 2, recvLen[i] * 2,
					 ncclFloat, i, comm->comm, comm->s);
		}
		ncclGroupEnd();
		cudaStreamSynchronize(comm->s);
//		printf("send compressed maxmin time: %.6f, len:%d\n", timer_stop(TMR_TMP), sendEnd[workerNum]);

//		timer_start(TMR_TMP);
//		de_convert<<<min(102400, recvIndex[workerNum]), feature_size>>>(buffer->recvValue, buffer->recvValue_Int, recvIndex[workerNum], feature_size, buffer->maxminRecv);
//		cudaDeviceSynchronize();
//		printf("decompress time: %.6f\n", timer_stop(TMR_TMP));
	} else {
		ncclGroupStart();
		for (int i = 0; i < workerNum; i++) {
			if (i == workerId) continue;
			ncclSend(buffer->sendValue + sendStart[i] * feature_size, (sendEnd[i] - sendStart[i]) * feature_size,
					 ncclFloat, i, comm->comm, comm->s);
			ncclRecv(buffer->recvValue + recvIndex[i] * feature_size, recvLen[i] * feature_size,
					 ncclFloat, i, comm->comm, comm->s);
		}
		ncclGroupEnd();
		cudaStreamSynchronize(comm->s);
//		printf("send Value time: %.6f, len:%d\n", timer_stop(TMR_TMP), (sendLen[0] + sendLen[1]) * feature_size);
	}

    int recvTotalLen = recvIndex[workerNum];
    delete []recvIndex;

    return recvTotalLen;
}

void GraphSum_sync_value_nccl(CUDAVariable* variables, Buffer* buffer, Comm* comm, Graph* graph, CacheBuffer* cacheBuffer, int feature_size, int workerId, int workerNum) {
//    timer_start(TMR_TMP);
    buffer_initIdx(buffer, workerNum);
    int localVertexSize = getLocalVertexSize(graph);

    if (batch_type == SEMI_BATCH) {
//        CUDACHECK(cudaDeviceSynchronize());
//        timer_start(TMR_TMP);
        cache_value<<<min(max_grid_size, cacheBuffer->cacheSize), feature_size>>>(variables->data, cacheBuffer->localValueCache,
											feature_size, cacheBuffer->cacheSize, cacheBuffer->cacheIndex2LocalID);
        CUDACHECK(cudaDeviceSynchronize());
//        printf("cache_value time: %.6f\n", timer_stop(TMR_TMP));
    }

    // mirror -> master
    CUDACHECK(cudaDeviceSynchronize());
    set_sync_value_mirror2master<<<min(max_grid_size, localVertexSize), feature_size>>>(buffer->sendID, buffer->sendValue, variables->data, feature_size, buffer->mirror2masterIdx,
                                                                                        graph->Mirror2Worker, graph->Local2Global, localVertexSize);
    CUDACHECK(cudaDeviceSynchronize());
//    CUDA_CHECK(cudaGetLastError());

    int recvLen = sendID_Value(buffer, buffer->mirror2masterStart, buffer->mirror2masterIdx, buffer->recvLen,
                               comm, feature_size, workerId, workerNum);
    mirror2master = recvLen;
    CUDACHECK(cudaDeviceSynchronize());
//	printf("Normal --- mirror2master send ok!, recvLen:%d\n", recvLen);
//    CUDA_CHECK(cudaGetLastError());
//	printf("sendID_Value ok!, recvLen:%d\n", recvLen);
//    timer_start(TMR_TMP);
    if (compress) {
        apply_sum_compressed<<<min(max_grid_size, recvLen), feature_size>>>(variables->data, buffer->recvID,
             buffer->maxminRecv, buffer->recvValue_Int, graph->Global2Local, recvLen, feature_size);
//        de_convert<<<min(102400, recvIndex[workerNum]), feature_size>>>(buffer->recvValue, buffer->recvValue_Int, recvIndex[workerNum], feature_size, buffer->maxminRecv);
    } else {
        apply_sum<<<min(max_grid_size, recvLen), feature_size>>>(variables->data, buffer->recvID, buffer->recvValue,
                                                                 graph->Global2Local,
                                                                 recvLen, feature_size);
    }
//	printf("apply_sum time: %.6f\n", timer_stop(TMR_TMP));
//	printf("apply_sum ok!\n");

    // master -> mirror
//    timer_start(TMR_TMP);
    set_sync_value_master2mirror<<<min(max_grid_size, localVertexSize), feature_size, sizeof(int) * workerNum>>>(buffer->sendID, buffer->sendValue, variables->data, feature_size, buffer->master2mirrorIdx,
                                                                                                                 graph->Master2Workers, graph->MasterWorkerIndex, graph->Local2Global, localVertexSize, workerNum);
    CUDACHECK(cudaDeviceSynchronize());
//	printf("set_sync_value_master2mirror time: %.6f\n", timer_stop(TMR_TMP));
//    cudaDeviceSynchronize();
    recvLen = sendID_Value(buffer, buffer->master2mirrorStart, buffer->master2mirrorIdx, buffer->recvLen,
                           comm, feature_size, workerId, workerNum);

    master2mirror = recvLen;
//	printf("Normal --- master2mirror send ok!, recvLen:%d\n", recvLen);
//    timer_start(TMR_TMP);
    if (compress) {
        apply_assign_compressed<<<min(max_grid_size, localVertexSize), feature_size>>>(variables->data, buffer->recvID,
                                                                                       buffer->maxminRecv, buffer->recvValue_Int, graph->Global2Local,
                                                                                       recvLen, feature_size);
    } else {
        apply_assign<<<min(max_grid_size, localVertexSize), feature_size>>>(variables->data, buffer->recvID,
                                                                            buffer->recvValue, graph->Global2Local,
                                                                            recvLen, feature_size);
    }
    CUDACHECK(cudaDeviceSynchronize());
//	printf("apply_assign time: %.6f\n", timer_stop(TMR_TMP));

    if (batch_type == SEMI_BATCH) {
        CUDACHECK(cudaDeviceSynchronize());
//        timer_start(TMR_TMP);
        cache_value<<<min(max_grid_size, cacheBuffer->cacheSize), feature_size>>>(variables->data, cacheBuffer->globalValueCache,
											 feature_size, cacheBuffer->cacheSize, cacheBuffer->cacheIndex2LocalID);
        CUDACHECK(cudaDeviceSynchronize());
//        printf("cache_value time: %.6f\n", timer_stop(TMR_TMP));
    }
}

void GraphSum_sync_value_nccl_active(CUDAVariable* variables, Buffer* buffer, Comm* comm, Graph* graph, int feature_size, int workerId, int workerNum, bool* active) {
//	timer_start(TMR_TMP);
    buffer_initIdx(buffer, workerNum);
    int localVertexSize = getLocalVertexSize(graph);
    // mirror -> master
    CUDACHECK(cudaDeviceSynchronize());
    set_sync_value_mirror2master_debug<<<min(max_grid_size, localVertexSize), feature_size>>>(buffer->sendID, buffer->sendValue, variables->data, feature_size, buffer->mirror2masterIdx,
                                 graph->Mirror2Worker, graph->Local2Global, localVertexSize, active);
    CUDACHECK(cudaDeviceSynchronize());
//    CUDA_CHECK(cudaGetLastError());

    int recvLen = sendID_Value(buffer, buffer->mirror2masterStart, buffer->mirror2masterIdx, buffer->recvLen,
                 comm, feature_size, workerId, workerNum);
    CUDACHECK(cudaDeviceSynchronize());
//    CUDA_CHECK(cudaGetLastError());
//	printf("sendID_Value ok!, recvLen:%d\n", recvLen);
//	timer_start(TMR_TMP);
    if (compress) {
        apply_sum_compressed<<<min(max_grid_size, recvLen), feature_size>>>(variables->data, buffer->recvID,
                                                                            buffer->maxminRecv, buffer->recvValue_Int, graph->Global2Local, recvLen, feature_size);
//        de_convert<<<min(102400, recvIndex[workerNum]), feature_size>>>(buffer->recvValue, buffer->recvValue_Int, recvIndex[workerNum], feature_size, buffer->maxminRecv);
    } else {
        apply_sum<<<min(max_grid_size, recvLen), feature_size>>>(variables->data, buffer->recvID, buffer->recvValue,
                                                                 graph->Global2Local,
                                                                 recvLen, feature_size);
    }
//	printf("apply_sum debug time: %.6f\n", timer_stop(TMR_TMP));
//	printf("apply_sum ok!\n");

    // master -> mirror
//	timer_start(TMR_TMP);
	set_sync_value_master2mirror_debug<<<min(max_grid_size, localVertexSize), feature_size, sizeof(int) * workerNum>>>(buffer->sendID, buffer->sendValue, variables->data, feature_size, buffer->master2mirrorIdx,
            graph->Master2Workers, graph->MasterWorkerIndex, graph->Local2Global, localVertexSize, workerNum, active);
    CUDACHECK(cudaDeviceSynchronize());
//	printf("set_sync_value_master2mirror time: %.6f\n", timer_stop(TMR_TMP));
//    cudaDeviceSynchronize();
    recvLen = sendID_Value(buffer, buffer->master2mirrorStart, buffer->master2mirrorIdx, buffer->recvLen,
                 comm, feature_size, workerId, workerNum);
//	timer_start(TMR_TMP);
    if (compress) {
        apply_assign_compressed<<<min(max_grid_size, localVertexSize), feature_size>>>(variables->data, buffer->recvID,
                                                                                       buffer->maxminRecv, buffer->recvValue_Int, graph->Global2Local,
                                                                                       recvLen, feature_size);
    } else {
        apply_assign<<<min(max_grid_size, localVertexSize), feature_size>>>(variables->data, buffer->recvID,
                                                                            buffer->recvValue, graph->Global2Local,
                                                                            recvLen, feature_size);
    }
    CUDACHECK(cudaDeviceSynchronize());
//	printf("apply_assign time: %.6f\n", timer_stop(TMR_TMP));
}

void GraphSum_sync_grad_nccl(CUDAVariable* variables, Buffer* buffer, Comm* comm, Graph* graph, CacheBuffer* cacheBuffer, int feature_size, int workerId, int workerNum) {
    buffer_initIdx(buffer, workerNum);
    int localVertexSize = getLocalVertexSize(graph);

    if (batch_type == SEMI_BATCH) {
//        CUDACHECK(cudaDeviceSynchronize());
//        timer_start(TMR_TMP);
        cache_value<<<min(max_grid_size, cacheBuffer->cacheSize), feature_size>>>(variables->grad, cacheBuffer->localGradCache,
             feature_size, cacheBuffer->cacheSize, cacheBuffer->cacheIndex2LocalID);
        CUDACHECK(cudaDeviceSynchronize());
    }

    // mirror -> master
    set_sync_value_mirror2master<<<min(max_grid_size, localVertexSize), feature_size>>>(buffer->sendID, buffer->sendValue, variables->grad, feature_size, buffer->mirror2masterIdx,
                                                                                              graph->Mirror2Worker, graph->Local2Global, localVertexSize);
    cudaDeviceSynchronize();
    int recvLen = sendID_Value(buffer, buffer->mirror2masterStart, buffer->mirror2masterIdx, buffer->recvLen,
                               comm, feature_size, workerId, workerNum);
    if (compress) {
        apply_sum_compressed<<<min(max_grid_size, recvLen), feature_size>>>(variables->grad, buffer->recvID,
                                                                            buffer->maxminRecv, buffer->recvValue_Int, graph->Global2Local, recvLen, feature_size);
//        de_convert<<<min(102400, recvIndex[workerNum]), feature_size>>>(buffer->recvValue, buffer->recvValue_Int, recvIndex[workerNum], feature_size, buffer->maxminRecv);
    } else {
        apply_sum<<<min(max_grid_size, recvLen), feature_size>>>(variables->grad, buffer->recvID, buffer->recvValue,
                                                                 graph->Global2Local,
                                                                 recvLen, feature_size);
    }
    CUDACHECK(cudaDeviceSynchronize());

    // master -> mirror
    set_sync_value_master2mirror<<<min(max_grid_size, localVertexSize), feature_size, sizeof(int) * workerNum>>>(buffer->sendID, buffer->sendValue, variables->grad, feature_size, buffer->master2mirrorIdx,
                                                                                                                       graph->Master2Workers, graph->MasterWorkerIndex, graph->Local2Global, localVertexSize, workerNum);
    cudaDeviceSynchronize();
    recvLen = sendID_Value(buffer, buffer->master2mirrorStart, buffer->master2mirrorIdx, buffer->recvLen,
                           comm, feature_size, workerId, workerNum);
    if (compress) {
        apply_assign_compressed<<<min(max_grid_size, localVertexSize), feature_size>>>(variables->grad, buffer->recvID,
                                          buffer->maxminRecv, buffer->recvValue_Int, graph->Global2Local,
                                                                            recvLen, feature_size);
    } else {
        apply_assign<<<min(max_grid_size, localVertexSize), feature_size>>>(variables->grad, buffer->recvID,
                                                                            buffer->recvValue, graph->Global2Local,
                                                                            recvLen, feature_size);
    }
    CUDACHECK(cudaDeviceSynchronize());

    if (batch_type == SEMI_BATCH) {
        CUDACHECK(cudaDeviceSynchronize());
//        timer_start(TMR_TMP);
        cache_value<<<min(max_grid_size, cacheBuffer->cacheSize), feature_size>>>(variables->grad, cacheBuffer->globalGradCache,
                              feature_size, cacheBuffer->cacheSize, cacheBuffer->cacheIndex2LocalID);
        CUDACHECK(cudaDeviceSynchronize());
//        printf("cache_value time: %.6f\n", timer_stop(TMR_TMP));
    }
}

void GraphSum_sync_grad_nccl_active(CUDAVariable* variables, Buffer* buffer, Comm* comm, Graph* graph, int feature_size, int workerId, int workerNum, bool *active) {
    buffer_initIdx(buffer, workerNum);
    int localVertexSize = getLocalVertexSize(graph);
    // mirror -> master
	set_sync_value_mirror2master_debug<<<min(max_grid_size, localVertexSize), feature_size>>>(buffer->sendID, buffer->sendValue, variables->grad, feature_size, buffer->mirror2masterIdx,
                                                                      graph->Mirror2Worker, graph->Local2Global, localVertexSize, active);
    cudaDeviceSynchronize();
    int recvLen = sendID_Value(buffer, buffer->mirror2masterStart, buffer->mirror2masterIdx, buffer->recvLen,
                               comm, feature_size, workerId, workerNum);

    if (compress) {
        apply_sum_compressed<<<min(max_grid_size, recvLen), feature_size>>>(variables->grad, buffer->recvID,
                                                                            buffer->maxminRecv, buffer->recvValue_Int, graph->Global2Local, recvLen, feature_size);
//        de_convert<<<min(102400, recvIndex[workerNum]), feature_size>>>(buffer->recvValue, buffer->recvValue_Int, recvIndex[workerNum], feature_size, buffer->maxminRecv);
    } else {
        apply_sum<<<min(max_grid_size, recvLen), feature_size>>>(variables->grad, buffer->recvID, buffer->recvValue,
                                                                 graph->Global2Local,
                                                                 recvLen, feature_size);
    }
    CUDACHECK(cudaDeviceSynchronize());

    // master -> mirror
	set_sync_value_master2mirror_debug<<<min(max_grid_size, localVertexSize), feature_size, sizeof(int) * workerNum>>>(buffer->sendID, buffer->sendValue, variables->grad, feature_size, buffer->master2mirrorIdx,
                graph->Master2Workers, graph->MasterWorkerIndex, graph->Local2Global, localVertexSize, workerNum, active);
    cudaDeviceSynchronize();
    recvLen = sendID_Value(buffer, buffer->master2mirrorStart, buffer->master2mirrorIdx, buffer->recvLen,
                           comm, feature_size, workerId, workerNum);
    if (compress) {
        apply_assign_compressed<<<min(max_grid_size, localVertexSize), feature_size>>>(variables->grad, buffer->recvID,
                                                                                       buffer->maxminRecv, buffer->recvValue_Int, graph->Global2Local,
                                                                                       recvLen, feature_size);
    } else {
        apply_assign<<<min(max_grid_size, localVertexSize), feature_size>>>(variables->grad, buffer->recvID,
                                                                            buffer->recvValue, graph->Global2Local,
                                                                            recvLen, feature_size);
    }
    CUDACHECK(cudaDeviceSynchronize());
}