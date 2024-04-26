#ifndef DRONE_CUDA_CC
#define DRONE_CUDA_CC

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

typedef struct CCValues {
    int *sendID, *recvID;
    int *sendCCValue, *recvCCValue;
    int *ccValue;
} CCValues;


Response CC_PEVal(Graph *g, CCValues* values, int workerId, int workerNum, Comm *comm);
Response CC_IncEVal(Graph *g, CCValues* values, int workerId, int workerNum, Comm *comm);

#endif //DRONE_CUDA_CC