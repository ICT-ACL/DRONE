#include "../common/timer.h"
#include "cuda_cache.cuh"
#include "cuda_runtime.h"
#include "cuda_compress.cuh"
#include "cuda_communicate.cuh"
#include "cuda_variable.cuh"
//#include "../common/timer.h"
//extern "C" {
#include "graph.h"
//}

__global__
void init_MirrorID_cache(int *index, int *cacheIndex2LocalID, int *localID2CacheIndex, int* Mirror2Worker, int localVertexSize) {
    int localID = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (localID < localVertexSize) {
        if (Mirror2Worker[localID] != -1) {
            int ind = atomicAdd(index, 1);
            cacheIndex2LocalID[ind] = localID;
			localID2CacheIndex[localID] = ind;
        }

        localID += stride;
    }
}

__global__
void init_MasterID_cache(int *index, int *cacheIndex2LocalID, int *localID2CacheIndex, int* MasterWorkerIndex, int localVertexSize) {
	int localID = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	while (localID < localVertexSize) {
		if (MasterWorkerIndex[localID + 1] - MasterWorkerIndex[localID] > 0) {
			int ind = atomicAdd(index, 1);
			cacheIndex2LocalID[ind] = localID;
			localID2CacheIndex[localID] = ind;
		}

		localID += stride;
	}
}

CacheBuffer::CacheBuffer(int cacheSize, int localVertexSize, int *Mirror2Worker, int *MasterWorkerIndex, int featureSize) : cacheSize(cacheSize) {
	int *index;
	cudaMallocManaged(&index, sizeof(int));
	*index = 0;

	cudaMalloc(&cacheIndex2LocalID, sizeof(int) * cacheSize);
	cudaMalloc(&localID2CacheIndex, sizeof(int) * localVertexSize);
//	init_local_ID_to_cache_index<<<min(102400, localVertexSize / 32 + 1), 32>>>(index, cacheIndex2LocalID, localID2CacheIndex,
//																				Mirror2Worker, MasterWorkerIndex, localVertexSize);
	init_MirrorID_cache<<<min(102400, localVertexSize / 32 + 1), 32>>>(index, cacheIndex2LocalID, localID2CacheIndex,
																				Mirror2Worker, localVertexSize);
	init_MasterID_cache<<<min(102400, localVertexSize / 32 + 1), 32>>>(index, cacheIndex2LocalID, localID2CacheIndex,
																	   MasterWorkerIndex, localVertexSize);
	CUDACHECK(cudaDeviceSynchronize());

	cudaMalloc(&localValueCache, sizeof(float) * cacheSize * featureSize);
	cudaMalloc(&globalValueCache, sizeof(float) * cacheSize * featureSize);
	cudaMalloc(&localGradCache, sizeof(float) * cacheSize * featureSize);
	cudaMalloc(&globalGradCache, sizeof(float) * cacheSize * featureSize);
	cudaMalloc(&differValue, sizeof(float) * cacheSize * featureSize);
	cudaMalloc(&maxAbsDiffer, sizeof(float) * cacheSize);
	cudaMalloc(&maxAbsValue, sizeof(float) * cacheSize);

	cudaFree(index);
    error_rate[0] = 0.1f;
    error_rate[1] = 0.0f;
}

CacheBuffer::~CacheBuffer() {
	cudaFree(localValueCache);
	cudaFree(globalValueCache);
	cudaFree(localGradCache);
	cudaFree(globalGradCache);
	cudaFree(differValue);
	cudaFree(maxAbsDiffer);
	cudaFree(maxAbsValue);
}

__global__
void cache_value(float *value, float *cacheValue, int feature_size, int cacheSize, int *cacheIndex2LocalID) {
    int cacheIdx = blockIdx.x;
    int stride = gridDim.x;

    while (cacheIdx < cacheSize) {
        int localId = cacheIndex2LocalID[cacheIdx];

        cacheValue[cacheIdx * feature_size + threadIdx.x] = value[localId * feature_size + threadIdx.x];
        cacheIdx += stride;
    }
}

__global__
void load_cache(float *value, float *cacheValue, int feature_size, int cacheSize, int *cacheIndex2LocalID) {
	int cacheIdx = blockIdx.x;
	int stride = gridDim.x;

	while (cacheIdx < cacheSize) {
		int localId = cacheIndex2LocalID[cacheIdx];

		value[localId * feature_size + threadIdx.x] = cacheValue[cacheIdx * feature_size + threadIdx.x];
		cacheIdx += stride;
	}
}

__global__
void cal_diff_absMax(float *localValueCache, int *cacheIndex2LocalID, float *values, float* differValue, int cacheSize, int featureSize,
                     float *maxAbsDiffer, float *maxAbsValue, int stride) {
    extern __shared__ float partial_max1[];
    float *partial_max2 = partial_max1 + stride;
    int cacheIdx = blockIdx.x;
    int gridSize = gridDim.x;
    int tid = threadIdx.x;

	float tmp_diff;

    while (cacheIdx < cacheSize) {
        int localId = cacheIndex2LocalID[cacheIdx] * featureSize + tid;
        int cacheId = cacheIdx * featureSize + tid;

        differValue[cacheId] = values[localId] - localValueCache[cacheId];
        partial_max1[tid] = fabsf(differValue[cacheId]);
        partial_max2[tid] = fabsf(values[localId]);
        if (tid < featureSize - stride) {
            differValue[cacheId + stride] = values[localId + stride] - localValueCache[cacheId + stride];
            partial_max1[tid] = fmaxf(partial_max1[tid], fabsf(differValue[cacheId + stride]));
            partial_max2[tid] = fmaxf(partial_max2[tid], fabsf(values[localId + stride]));
        }
//		tmp_diff = values[localId] - localValueCache[cacheId];
//		partial_max1[tid] = fabsf(tmp_diff);
//		partial_max2[tid] = fabsf(values[localId]);
//		if (tid < featureSize - stride) {
//			tmp_diff = values[localId + stride] - localValueCache[cacheId + stride];
//			partial_max1[tid] = fmaxf(partial_max1[tid], fabsf(tmp_diff));
//			partial_max2[tid] = fmaxf(partial_max2[tid], fabsf(values[localId + stride]));
//		}
        __syncthreads();
        int stride_ = stride >> 1;
        if (tid < stride_) {
            volatile float *vol_max1 = partial_max1;
            volatile float *vol_max2 = partial_max2;
            while (stride_ > 0) {
                vol_max1[tid] = fmaxf(vol_max1[tid], vol_max1[tid + stride_]);
                vol_max2[tid] = fmaxf(vol_max2[tid], vol_max2[tid + stride_]);
                stride_ >>= 1;
            }
        }
        if (tid == 0) {
            maxAbsDiffer[cacheIdx] = partial_max1[0];
            maxAbsValue[cacheIdx] = partial_max2[0];
        }

        cacheIdx += gridSize;
    }
}

__global__
void cal_diff(float *cache, int *cacheIndex2LocalID, float *value, float *differ, int cacheSize, int featureSize) {
	int cacheIdx = blockIdx.x;
	int stride = gridDim.x;

	while (cacheIdx < cacheSize) {
		int localId = cacheIndex2LocalID[cacheIdx];

		differ[cacheIdx * featureSize + threadIdx.x] = value[localId * featureSize + threadIdx.x] - cache[cacheIdx * featureSize + threadIdx.x];
		cacheIdx += stride;
	}
}

__global__
void update_localCache(float *localCache, int *cacheIndex2LocalID, float *differ, int cacheSize, int featureSize, bool *active) {
	int cacheIdx = blockIdx.x;
	int stride = gridDim.x;

	while (cacheIdx < cacheSize) {
		int localId = cacheIndex2LocalID[cacheIdx];
		if (active[localId]) {
			localCache[cacheIdx * featureSize + threadIdx.x] += differ[cacheIdx * featureSize + threadIdx.x];
		}
		cacheIdx += stride;
	}
}

__global__
void cal_abs_max(float *value, int *cacheIndex2LocalID, float *maxAbsValue, int cacheSize, int featureSize, const int stride, bool useLocalID) {
	extern __shared__ float partial_max[];
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	while (bid < cacheSize) {
		int dstID = bid * featureSize + tid;
		if (useLocalID) dstID = cacheIndex2LocalID[bid] * featureSize + tid;

//		if (bid == 0 && tid == 0) {
//			printf("dstID:%d\n", dstID);
//		}

		partial_max[tid] = fabsf(value[dstID]);
		if (tid < featureSize - stride) {
			partial_max[tid] = fmaxf(partial_max[tid], fabsf(value[dstID + stride]));
		}
		int stride_ = stride >> 1;

		__syncthreads();
		if (tid < stride_) {
			volatile float *vol_max = partial_max;
			while (stride_ > 0) {
				vol_max[tid] = fmaxf(vol_max[tid], vol_max[tid + stride_]);
				stride_ >>= 1;
			}
		}
		if (tid == 0) maxAbsValue[bid] = partial_max[0];
		bid += gridDim.x;
	}
}

__global__
void cal_active(float *maxAbsDiffer, float *maxAbsValue, bool *active, int *cacheIndex2LocalID, int cacheSize, float error_rate) {
    int cacheID = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (cacheID < cacheSize) {
        active[cacheIndex2LocalID[cacheID]] = maxAbsDiffer[cacheID] > maxAbsValue[cacheID] * error_rate;

        cacheID += stride;
    }
}

__global__
void set_sync_mirror2master_cache(int* sendID, float* sendValue, float* diffValue, int feature_size, int *cacheIndex2LocalID,
                                        int* mirror2masterIdx, int* Mirror2Worker, int* Local2Global, int mirrorSize, bool *active) {
    int stride = gridDim.x;
    int cacheID = blockIdx.x;
    __shared__ int idx[1];
    while (cacheID < mirrorSize) {
		int localID = cacheIndex2LocalID[cacheID];
        if (!active[localID]) {
			cacheID += stride;
            continue;
        }

        if (threadIdx.x == 0) {
            idx[0] = atomicAdd(mirror2masterIdx + Mirror2Worker[localID], 1);
//			idx[0] = mirror2masterIdx[Mirror2Worker[localID]]++;
            sendID[idx[0]] = Local2Global[localID];
            idx[0] *= feature_size;
        }
        __syncthreads();
        sendValue[idx[0] + threadIdx.x] = diffValue[cacheID * feature_size + threadIdx.x];

		cacheID += stride;
    }
}

__global__
void set_sync_master2mirror_cache(int* sendID, float* sendValue, float* globalCache, float *diffValue, int feature_size, int *cacheIndex2LocalID, int* master2mirrorIdx,
								  int* Master2Workers, int *MasterWorkerIndex, int* Local2Global, int masterStart, int masterEnd, int workerNum, bool *active) {
	extern __shared__ int idx[];
	int cacheID = blockIdx.x + masterStart;
	int stride = gridDim.x;
	int tid = threadIdx.x;

	while (cacheID < masterEnd) {
		int localID = cacheIndex2LocalID[cacheID];
		if (active[localID]) {
			globalCache[cacheID * feature_size + tid] += diffValue[cacheID * feature_size + tid];

			const int searchLen = MasterWorkerIndex[localID + 1] - MasterWorkerIndex[localID];
			int mirrorId = tid;
			while (mirrorId < searchLen) {
				int worker = Master2Workers[mirrorId + MasterWorkerIndex[localID]];
				idx[mirrorId] = atomicAdd(master2mirrorIdx + worker, 1);
				sendID[idx[mirrorId]] = Local2Global[localID];
				mirrorId += blockDim.x;
			}
			if (tid == 0) idx[workerNum - 1] = searchLen;
			__syncthreads();
			for (int i = 0; i < idx[workerNum - 1]; i++) {
				sendValue[idx[i] * feature_size + tid] = globalCache[cacheID * feature_size + tid];
			}
		}

		cacheID += stride;
	}
}

__global__
void apply_sum_diff_compressed(float* globalCache, int* recvID, float* maxmin, uint8_t* recvIn, int* Global2Local,
							   int recvLen, int feature_size, int *localID2CacheIndex, bool *active) {
	int id = blockIdx.x;
	int stride = gridDim.x;
	int tid = threadIdx.x;

	while (id < recvLen) {
		float max_ = maxmin[2 * id];
		float min_ = maxmin[2 * id + 1];

		int localID = Global2Local[recvID[id]];
		int cacheID = localID2CacheIndex[localID];
		float val = min_ + (max_ - min_) * recvIn[id * feature_size + tid] / 255;
		atomicAdd(globalCache + cacheID * feature_size + tid, val);
		id += stride;
		active[localID] = true;
	}
}

__global__
void apply_sum_diff_nocompressed(float* globalCache, int* recvID, float* recvIn, int* Global2Local,
                               int recvLen, int feature_size, int *localID2CacheIndex, bool *active) {
    int id = blockIdx.x;
    int stride = gridDim.x;
    int tid = threadIdx.x;

    while (id < recvLen) {
        int localID = Global2Local[recvID[id]];
        int cacheID = localID2CacheIndex[localID];
        float val = recvIn[id * feature_size + tid];
        atomicAdd(globalCache + cacheID * feature_size + tid, val);
        id += stride;
        active[localID] = true;
    }
}

__global__
void apply_assign_global_compressed(float* globalCache, int* recvID, float* maxmin, uint8_t* recvIn, int* Global2Local,
									int recvLen, int feature_size, int *localID2CacheIndex) {
	int id = blockIdx.x;
	int stride = gridDim.x;
	int tid = threadIdx.x;

	while (id < recvLen) {
		float max_ = maxmin[2 * id];
		float min_ = maxmin[2 * id + 1];

		int localID = Global2Local[recvID[id]];
		int cacheID = localID2CacheIndex[localID];
		float val = min_ + (max_ - min_) * recvIn[id * feature_size + tid] / 255;
		globalCache[cacheID * feature_size + tid] = val;
		id += stride;
	}
}

__global__
void apply_assign_global_nocompressed(float* globalCache, int* recvID, float* recvIn, int* Global2Local,
                                    int recvLen, int feature_size, int *localID2CacheIndex) {
    int id = blockIdx.x;
    int stride = gridDim.x;
    int tid = threadIdx.x;

    while (id < recvLen) {
        int localID = Global2Local[recvID[id]];
        int cacheID = localID2CacheIndex[localID];
        globalCache[cacheID * feature_size + tid] = recvIn[id * feature_size + tid];
        id += stride;
    }
}


void GraphSum_sync_value_nccl_cache(CUDAVariable* variables, Buffer* buffer, Comm* comm, Graph* graph, CacheBuffer* cacheBuffer,
									int feature_size, int workerId, int workerNum, bool *active, int layer) {
//	printf("call GraphSum_sync_value_nccl_cache!\n");
    timer_start(TMR_TMP_);
    buffer_initIdx(buffer, workerNum);
    int localVertexSize = getLocalVertexSize(graph);

	int stride;
	for (int k = 0; k < 32; k++) {
		if ((feature_size - 1) >> k == 1) {
			stride = 1 << k;
			break;
		}
	}
    assert(stride <= 32);

    cal_diff_absMax<<<min(max_grid_size, cacheBuffer->cacheSize), stride, sizeof(float) * stride * 2>>>
    (cacheBuffer->localValueCache, cacheBuffer->cacheIndex2LocalID, variables->data, cacheBuffer->differValue, cacheBuffer->cacheSize, feature_size,
     cacheBuffer->maxAbsDiffer, cacheBuffer->maxAbsValue, stride);
    CUDACHECK(cudaDeviceSynchronize());
    float tmp = timer_stop(TMR_TMP_);
    if (time_debug) printf("Worker %d --- GraphSum_sync_value_nccl_cache cal_diff_absMax time:%.6f\n", workerId, tmp);


	timer_start(TMR_TMP_);
    cal_active<<<min(max_grid_size, cacheBuffer->cacheSize / 512 + 1), 512>>>(cacheBuffer->maxAbsDiffer, cacheBuffer->maxAbsValue,
                                                          active, cacheBuffer->cacheIndex2LocalID, cacheBuffer->cacheSize, cacheBuffer->error_rate[0]);
	CUDACHECK(cudaDeviceSynchronize());
	tmp = timer_stop(TMR_TMP_);
	if (time_debug) printf("Worker %d --- GraphSum_sync_value_nccl_cache cal_active time:%.6f\n", workerId, tmp);

	timer_start(TMR_TMP_);
    set_sync_mirror2master_cache<<<min(max_grid_size, graph->MirrorSize), feature_size>>>
	(buffer->sendID, buffer->sendValue, cacheBuffer->differValue, feature_size, cacheBuffer->cacheIndex2LocalID,
	 buffer->mirror2masterIdx, graph->Mirror2Worker, graph->Local2Global, graph->MirrorSize, active);
    CUDACHECK(cudaDeviceSynchronize());
	tmp = timer_stop(TMR_TMP_);
	if (time_debug) printf("Worker %d --- GraphSum_sync_value_nccl_cache set_sync_mirror2master_cache time:%.6f\n", workerId, tmp);

	timer_start(TMR_TMP_);
    int recvLen = sendID_Value(buffer, buffer->mirror2masterStart, buffer->mirror2masterIdx, buffer->recvLen,
                               comm, feature_size, workerId, workerNum);
	tmp = timer_stop(TMR_TMP_);
    printf("Worker %d --- GraphSum_sync_value_nccl_cache sendID_Value mirror2master time:%.6f, recvLen:%d / %d\n", workerId, tmp, recvLen, mirror2master);
	if (time_debug) printf("Worker %d --- GraphSum_sync_value_nccl_cache sendID_Value mirror2master time:%.6f\n", workerId, tmp);
	mirror2master_forward_total[layer] += recvLen;

	int sendLen = 0;
	for (int i = 0; i < workerNum; i++) sendLen += buffer->mirror2masterIdx[i] - buffer->mirror2masterStart[i];
//	printf("forward, layer:%d, percentage: %.8f, error_rate:%.8f\n", layer, 1.0 * sendLen / graph->MirrorSize, cacheBuffer->error_rate[0]);
//    if (cacheBuffer->error_rate[0] > 0.03 && cacheBuffer->error_rate[0] < 0.5) {
//        if (sendLen < 1.0 * current_epoch / 1000 * graph->MirrorSize) cacheBuffer->error_rate[0] *= 0.9;
//        if (sendLen > 1.0 * current_epoch / 800 * graph->MirrorSize) cacheBuffer->error_rate[0] *= 1.1;
//    }

    if (update_threshold) {
        cacheBuffer->error_rate[0] *= 0.9;
    }

    CUDACHECK(cudaDeviceSynchronize());
//	printf("ok4!\n");
//    CUDA_CHECK(cudaGetLastError());
//	printf("Cache --- mirror2master send feature ok!, recvLen:%d\n", recvLen);
//    timer_start(TMR_TMP);
	timer_start(TMR_TMP_);
    if (recvLen > 0) {
        if (compress) {
            // 根据mirror发的消息完成更新global
            apply_sum_diff_compressed<<<min(max_grid_size, recvLen), feature_size>>>(cacheBuffer->globalValueCache,
                                                                                     buffer->recvID,
                                                                                     buffer->maxminRecv,
                                                                                     buffer->recvValue_Int,
                                                                                     graph->Global2Local, recvLen,
                                                                                     feature_size,
                                                                                     cacheBuffer->localID2CacheIndex,
                                                                                     active);
        } else {
            apply_sum_diff_nocompressed<<<min(max_grid_size, recvLen), feature_size>>>(cacheBuffer->globalValueCache,
                                                                                     buffer->recvID,
                                                                                     buffer->recvValue,
                                                                                     graph->Global2Local, recvLen,
                                                                                     feature_size,
                                                                                     cacheBuffer->localID2CacheIndex,
                                                                                     active);
        }
    }
	CUDACHECK(cudaDeviceSynchronize());
	tmp = timer_stop(TMR_TMP_);
	if (time_debug) printf("Worker %d --- GraphSum_sync_value_nccl_cache apply_sum_diff_compressed time:%.6f\n", workerId, tmp);

	timer_start(TMR_TMP_);
	// 如果active，更新自身localCache (包括local和global)
	update_localCache<<<min(max_grid_size, cacheBuffer->cacheSize), feature_size>>>(cacheBuffer->localValueCache,
				cacheBuffer->cacheIndex2LocalID, cacheBuffer->differValue, cacheBuffer->cacheSize, feature_size, active);
	CUDACHECK(cudaDeviceSynchronize());
	tmp = timer_stop(TMR_TMP_);
	if (time_debug) printf("Worker %d --- GraphSum_sync_value_nccl_cache update_localCache time:%.6f\n", workerId, tmp);


    // master -> mirror
//    timer_start(TMR_TMP);
	// 设置master->mirror的待发送消息，顺带更新master的globalValue
//    set_sync_value_master2mirror<<<min(max_grid_size, localVertexSize), feature_size, sizeof(int) * workerNum>>>(buffer->sendID, buffer->sendValue, variables->data, feature_size, buffer->master2mirrorIdx,
//    graph->Master2Workers, graph->MasterWorkerIndex, graph->Local2Global, localVertexSize, workerNum);
	timer_start(TMR_TMP_);
	set_sync_master2mirror_cache<<<min(max_grid_size, graph->MasterSize), feature_size, sizeof(int) * workerNum>>>
			(buffer->sendID, buffer->sendValue, cacheBuffer->globalValueCache, cacheBuffer->differValue, feature_size, cacheBuffer->cacheIndex2LocalID,
			 buffer->master2mirrorIdx, graph->Master2Workers, graph->MasterWorkerIndex, graph->Local2Global, graph->MirrorSize, cacheBuffer->cacheSize,
			 workerNum, active);
	CUDACHECK(cudaDeviceSynchronize());
	tmp = timer_stop(TMR_TMP_);
	if (time_debug) printf("Worker %d --- GraphSum_sync_value_nccl_cache set_sync_master2mirror_cache time:%.6f\n", workerId, tmp);

	timer_start(TMR_TMP_);
    recvLen = sendID_Value(buffer, buffer->master2mirrorStart, buffer->master2mirrorIdx, buffer->recvLen,
                           comm, feature_size, workerId, workerNum);
	tmp = timer_stop(TMR_TMP_);
    printf("Worker %d --- GraphSum_sync_value_nccl_cache sendID_Value master2mirror time:%.6f, recvLen:%d / %d\n", workerId, tmp, recvLen, master2mirror);
	if (time_debug) printf("Worker %d --- GraphSum_sync_value_nccl_cache sendID_Value master2mirror time:%.6f\n", workerId, tmp);
	master2mirror_forward_total[layer] += recvLen;

	timer_start(TMR_TMP_);
    if (recvLen > 0) {
        if (compress) {
            apply_assign_global_compressed<<<min(max_grid_size, recvLen), feature_size>>>(
                    cacheBuffer->globalValueCache, buffer->recvID,
                    buffer->maxminRecv, buffer->recvValue_Int, graph->Global2Local, recvLen, feature_size,
                    cacheBuffer->localID2CacheIndex);
        } else {
            apply_assign_global_nocompressed<<<min(max_grid_size, recvLen), feature_size>>>(
                    cacheBuffer->globalValueCache, buffer->recvID, buffer->recvValue,
                    graph->Global2Local, recvLen, feature_size, cacheBuffer->localID2CacheIndex);
        }
    }
    CUDACHECK(cudaDeviceSynchronize());
	tmp = timer_stop(TMR_TMP_);
	if (time_debug) printf("Worker %d --- GraphSum_sync_value_nccl_cache apply_assign_global_compressed time:%.6f\n", workerId, tmp);

	timer_start(TMR_TMP_);
	load_cache<<<min(max_grid_size, cacheBuffer->cacheSize), feature_size>>>(variables->data, cacheBuffer->globalValueCache,
																			 feature_size, cacheBuffer->cacheSize, cacheBuffer->cacheIndex2LocalID);
	CUDACHECK(cudaDeviceSynchronize());
	tmp = timer_stop(TMR_TMP_);
	if (time_debug) printf("Worker %d --- GraphSum_sync_value_nccl_cache load_cache time:%.6f\n", workerId, tmp);
}

void GraphSum_sync_grad_nccl_cache(CUDAVariable* variables, Buffer* buffer, Comm* comm, Graph* graph, CacheBuffer* cacheBuffer,
                                    int feature_size, int workerId, int workerNum, bool *active, int layer) {
//    timer_start(TMR_TMP);
    buffer_initIdx(buffer, workerNum);
    int localVertexSize = getLocalVertexSize(graph);

    // 分别计算最大的误差和最大的绝对值
    int stride;
    for (int k = 0; k < 32; k++) {
        if ((feature_size - 1) >> k == 1) {
            stride = 1 << k;
            break;
        }
    }
	cal_diff_absMax<<<min(max_grid_size, cacheBuffer->cacheSize), stride, sizeof(float) * stride * 2>>>
			(cacheBuffer->localGradCache, cacheBuffer->cacheIndex2LocalID, variables->grad, cacheBuffer->differValue, cacheBuffer->cacheSize, feature_size,
			 cacheBuffer->maxAbsDiffer, cacheBuffer->maxAbsValue, stride);


    cal_active<<<min(max_grid_size, cacheBuffer->cacheSize / 32 + 1), 32>>>(cacheBuffer->maxAbsDiffer, cacheBuffer->maxAbsValue,
                                     active, cacheBuffer->cacheIndex2LocalID, cacheBuffer->cacheSize, cacheBuffer->error_rate[1]);
//	CUDACHECK(cudaDeviceSynchronize());
//	CUDA_CHECK(cudaGetLastError());
//	printf("ok4!\n");

    CUDACHECK(cudaDeviceSynchronize());
    set_sync_mirror2master_cache<<<min(max_grid_size, graph->MirrorSize), feature_size>>>
            (buffer->sendID, buffer->sendValue, cacheBuffer->differValue, feature_size, cacheBuffer->cacheIndex2LocalID,
             buffer->mirror2masterIdx, graph->Mirror2Worker, graph->Local2Global, graph->MirrorSize, active);
    CUDACHECK(cudaDeviceSynchronize());
//    CUDA_CHECK(cudaGetLastError());
//	printf("ok5!\n");

    int recvLen = sendID_Value(buffer, buffer->mirror2masterStart, buffer->mirror2masterIdx, buffer->recvLen,
                               comm, feature_size, workerId, workerNum);
	mirror2master_backward_total[layer] += recvLen;

	int sendLen = 0;
	for (int i = 0; i < workerNum; i++) sendLen += buffer->mirror2masterIdx[i] - buffer->mirror2masterStart[i];
//	printf("backward, layer:%d, percentage: %.8f, error_rate:%.8f\n", layer, 1.0 * sendLen / graph->MirrorSize, cacheBuffer->error_rate[1]);
//	if (sendLen < 0.3 * graph->MirrorSize) cacheBuffer->error_rate[1] *= 0.9;
//	if (sendLen > 0.4 * graph->MirrorSize) cacheBuffer->error_rate[1] *= 1.1;
	CUDACHECK(cudaDeviceSynchronize());
//	printf("ok4!\n");
//    CUDA_CHECK(cudaGetLastError());
//    printf("Cache --- mirror2master send grad ok!, recvLen:%d\n", recvLen);
//    timer_start(TMR_TMP);
    CUDACHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    if (recvLen > 0) {
        if (compress) {
            // 根据mirror发的消息完成更新globalCache
            apply_sum_diff_compressed<<<min(max_grid_size, recvLen), feature_size>>>(cacheBuffer->globalGradCache,
                                                                                     buffer->recvID,
                                                                                     buffer->maxminRecv,
                                                                                     buffer->recvValue_Int,
                                                                                     graph->Global2Local, recvLen,
                                                                                     feature_size,
                                                                                     cacheBuffer->localID2CacheIndex,
                                                                                     active);
        } else {
            apply_sum_diff_nocompressed<<<min(max_grid_size, recvLen), feature_size>>>(cacheBuffer->globalGradCache,
                                                                                     buffer->recvID,
                                                                                     buffer->recvValue,
                                                                                     graph->Global2Local, recvLen,
                                                                                     feature_size,
                                                                                     cacheBuffer->localID2CacheIndex,
                                                                                     active);
        }
    }
    CUDACHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    //	printf("apply_sum time: %.6f\n", timer_stop(TMR_TMP));
    //	printf("apply_sum ok!\n");

    // 如果active，更新自身localCache
    update_localCache<<<min(max_grid_size, cacheBuffer->cacheSize), feature_size>>>(cacheBuffer->localGradCache,
                cacheBuffer->cacheIndex2LocalID, cacheBuffer->differValue, cacheBuffer->cacheSize, feature_size, active);


    // master -> mirror
//    timer_start(TMR_TMP);
    // 设置master->mirror的待发送消息，顺带更新master的globalGradCache
    set_sync_master2mirror_cache<<<min(max_grid_size, graph->MasterSize), feature_size, sizeof(int) * workerNum>>>
            (buffer->sendID, buffer->sendValue, cacheBuffer->globalGradCache, cacheBuffer->differValue, feature_size, cacheBuffer->cacheIndex2LocalID,
             buffer->master2mirrorIdx, graph->Master2Workers, graph->MasterWorkerIndex, graph->Local2Global, graph->MirrorSize, cacheBuffer->cacheSize,
             workerNum, active);
    CUDACHECK(cudaDeviceSynchronize());
//	printf("set_sync_value_master2mirror time: %.6f\n", timer_stop(TMR_TMP));
//    cudaDeviceSynchronize();
    recvLen = sendID_Value(buffer, buffer->master2mirrorStart, buffer->master2mirrorIdx, buffer->recvLen,
                           comm, feature_size, workerId, workerNum);
	master2mirror_backward_total[layer] += recvLen;

//    printf("Cache --- master2mirror send grad ok!, recvLen:%d\n", recvLen);
//    timer_start(TMR_TMP);
    if (recvLen > 0) {
        if (compress) {
            apply_assign_global_compressed<<<min(max_grid_size, recvLen), feature_size>>>(cacheBuffer->globalGradCache,
                                                                                          buffer->recvID,
                                                                                          buffer->maxminRecv,
                                                                                          buffer->recvValue_Int,
                                                                                          graph->Global2Local, recvLen,
                                                                                          feature_size,
                                                                                          cacheBuffer->localID2CacheIndex);
        } else {
            apply_assign_global_nocompressed<<<min(max_grid_size, recvLen), feature_size>>>(cacheBuffer->globalGradCache,
                                                                                          buffer->recvID,
                                                                                          buffer->recvValue,
                                                                                          graph->Global2Local, recvLen,
                                                                                          feature_size,
                                                                                          cacheBuffer->localID2CacheIndex);
        }
    }
    CUDACHECK(cudaDeviceSynchronize());
//	printf("apply_assign time: %.6f\n", timer_stop(TMR_TMP));
    load_cache<<<min(max_grid_size, cacheBuffer->cacheSize), feature_size>>>(variables->grad, cacheBuffer->globalGradCache,
                                    feature_size, cacheBuffer->cacheSize, cacheBuffer->cacheIndex2LocalID);
}