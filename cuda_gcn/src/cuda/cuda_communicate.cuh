#ifndef CUDA_COMMUNICATE_CUH
#define CUDA_COMMUNICATE_CUH

#include "cuda_variable.cuh"
#include "cuda_runtime.h"
#include "cuda_compress.cuh"
#include "cstdint"
//#include "../common/timer.h"
//extern "C" {
#include "graph.h"
//}
//#include <stdio.h>
#include <vector>
using std::vector;

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

struct Buffer {
	float *sendValue, *recvValue; // device
	int *sendID, *recvID; // device
	uint8_t *sendValue_Int, *recvValue_Int;
    float *value;
	float *maxminSend;
	float *maxminRecv;
	int *mirror2masterLen, *master2mirrorLen; // vertex num, device and host
	int *mirror2masterIdx, *master2mirrorIdx; // host
	int *mirror2masterStart, *master2mirrorStart;
    int *recvLen;
	int maxDim;
};

class CacheBuffer {
public:
	float *localValueCache, *globalValueCache, *localGradCache, *globalGradCache;
	int cacheSize;
	int *cacheIndex2LocalID, *localID2CacheIndex;
	float *differValue;
	float *maxAbsDiffer, *maxAbsValue;
	float error_rate[2];
	CacheBuffer(int cacheSize, int localVertexSize, int* Mirror2Worker, int* MasterWorkerIndex, int featureSize);
	~CacheBuffer();
};

void buffer_init(Buffer* buffer, Graph* graph, int workerId, int workerNum);

void delete_buffer(Buffer* buffer);

void buffer_initIdx(Buffer* buffer, int workerNum);

void weight_broadcast_value_nccl(vector<CUDAVariable> * variables, Comm* comm);

void weight_reduce_grad_nccl(vector<CUDAVariable>* variables, Comm* comm);

void GraphSum_sync_value_nccl_active(CUDAVariable* variables, Buffer* buffer, Comm* comm, Graph* graph, int feature_size, int workerId, int workerNum, bool *active);
void GraphSum_sync_value_nccl(CUDAVariable* variables, Buffer* buffer, Comm* comm, Graph* graph, CacheBuffer* cacheBuffer, int feature_size, int workerId, int workerNum);

void GraphSum_sync_grad_nccl(CUDAVariable* variables, Buffer* buffer, Comm* comm, Graph* graph, CacheBuffer* cacheBuffer, int feature_size, int workerId, int workerNum);
void GraphSum_sync_grad_nccl_active(CUDAVariable* variables, Buffer* buffer, Comm* comm, Graph* graph, int feature_size, int workerId, int workerNum, bool *active);

int sendID_Value(Buffer* buffer, int* sendStart, int* sendEnd, int *recvLen, Comm* comm, int feature_size, int workerId, int workerNum);

#endif