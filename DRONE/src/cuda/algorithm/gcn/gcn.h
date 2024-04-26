#ifndef DRONE_CUDA_GCN
#define DRONE_CUDA_GCN

#include "../../graph.h"
#include "../common.h"

typedef void* GCNHandle;

//Handle create_handle(int size);
//void* get_data(Handle handle, int index);
//void free_handle(Handle handle);


typedef struct GCNValues {
    int *sendID, *recvID;
    float *sendDiff, *recvDiff;
    int *in_degree;
    GCNHandle handle;
} GCNValues;

void loadOutDegree(Graph *g, PRValues* values, int *globalOutDegree);

Response PR_PEVal(Graph *g, PRValues* values, int workerId, int workerNum, Comm *comm);
Response PR_IncEVal(Graph *g, PRValues* values, int workerId, int workerNum, Comm *comm);

#endif //DRONE_CUDA_PR