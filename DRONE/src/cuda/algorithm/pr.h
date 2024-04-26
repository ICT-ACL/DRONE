#ifndef DRONE_CUDA_PR
#define DRONE_CUDA_PR

#include "../graph.h"
#include "common.h"

typedef struct PRValues {
    int *sendID, *recvID;
    float *sendDiff, *recvDiff;
    float *prValue, *accValue, *diffValue;
    float *globalAcc, *globalDiff;
    int *outDegree;
    // prValue: 保存pr值
    // accValue: 保存累计值，不清空，不显示同步
    // diffValue: 每次更新得时候判断，若超过minerr，则更新该值。在发送消息时，使用sum同步
} PRValues;

void loadOutDegree(Graph *g, PRValues* values, int *globalOutDegree);

Response PR_PEVal(Graph *g, PRValues* values, int workerId, int workerNum, Comm *comm);
Response PR_IncEVal(Graph *g, PRValues* values, int workerId, int workerNum, Comm *comm);

#endif //DRONE_CUDA_PR