#ifndef DRONE_CUDA_SSSP
#define DRONE_CUDA_SSSP

#include "../graph.h"
#include "common.h"
//#include "nccl.h"

//#ifdef __cplusplus
//#define cplus __cplusplus
//#undef __cplusplus
//#endif
//#include "nccl.h"
//#ifdef cplus
//#define __cplusplus cplus
//#undef cplus
//#endif

typedef struct SSSPValues {
    int *sendID, *recvID;
    float *sendDis, *recvDis;
    float *distance;
} SSSPValues;


Response SSSP_PEVal(Graph *g, SSSPValues* values, int startId, int workerId, int workerNum, Comm *comm);
Response SSSP_IncEVal(Graph *g, SSSPValues* values, int workerId, int workerNum, Comm *comm);

#endif //DRONE_CUDA_SSSP