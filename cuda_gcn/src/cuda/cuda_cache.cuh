#ifndef CUDA_CACHE_CUH
#define CUDA_CACHE_CUH
#include "../common/utils.h"
#include "cuda_communicate.cuh"

//__global__
//void init_local_ID_to_cache_index(int *index, int *cacheIndex2LocalID, int *localID2CacheIndex, int* Mirror2Worker, int* MasterWorkerIndex, int localVertexSize);

__global__
void init_MirrorID_cache(int *index, int *cacheIndex2LocalID, int *localID2CacheIndex, int* Mirror2Worker, int localVertexSize);

__global__
void init_MasterID_cache(int *index, int *cacheIndex2LocalID, int *localID2CacheIndex, int* MasterWorkerIndex, int localVertexSize);

__global__
void cache_value(float *value, float *cacheValue, int feature_size, int cacheSize, int *cacheIndex2LocalID);


void GraphSum_sync_value_nccl_cache(CUDAVariable* variables, Buffer* buffer, Comm* comm, Graph* graph, CacheBuffer* cacheBuffer,
									int feature_size, int workerId, int workerNum, bool *active, int layer);

void GraphSum_sync_grad_nccl_cache(CUDAVariable* variables, Buffer* buffer, Comm* comm, Graph* graph, CacheBuffer* cacheBuffer,
                                   int feature_size, int workerId, int workerNum, bool *active, int layer);
#endif