#ifndef DRONE_CUDA_COMMON
#define DRONE_CUDA_COMMON

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

typedef struct Response {
    int Mirror2MasterSendSize, Mirror2MasterRecvSize, Master2MirrorSendSize, Master2MirrorRecvSize;
    int VisitedSize, LocalVertexSize;
    float CalTime, SendTime;
}Response;

int countActive(int *active, int localVertexSize, int gridSize, int blockSize);

__global__ void CUDA_FILL_INT(int *array, int val, int maxLen);
__global__ void CUDA_FILL_float(float *array, float val, int maxLen);

#endif