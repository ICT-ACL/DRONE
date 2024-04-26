#ifndef DRONE_CUDA_GRAPH
#define DRONE_CUDA_GRAPH

//#include "nccl.h"

#ifdef __cplusplus
#define cplus __cplusplus
#undef __cplusplus
#endif
#include "nccl.h"
#ifdef cplus
#define __cplusplus cplus
#undef cplus
#endif

typedef struct Graph {
    int *Global2Local; // global id to local id
    int *Local2Global; // local id to global id

    int *MasterWorkerIndex; //  master local id -> mirror worker location list in CSR format
    int *Master2Workers;
    int MasterSize, MirrorWorkerSize;

    int *Mirror2Worker; // mirror local id -> master worker location
    int MirrorSize;

    int *index; // local id -> dst index
    int *dst; // dst index -> dst local id
    float *edgeLen;

    int *localVertexSize, *globalVertexSize, *edgeSize;

    int gridSize, blockSize;

    int *active;
    int *updateByMessage;
    int GID;
} Graph;

typedef struct Comm {
    ncclComm_t comm;
    cudaStream_t s;
} Comm;

Graph* build_graph(int globalVertexSize, int edgeSize, int *u, int *v, int workerId, int workerNum, Comm *comm);

int getLocalVertexSize(Graph* g);
int getGlobalVertexSize(Graph* g);

//the masterVertex and mirrorVertex need to be deduplicated before invoke these functions
void addMasterRoute(Graph* g, int *masterVertex, int *mirrorNumber, int *mirrorWorkers, int masterSize, int mirrorWorkerSize);
void addMirrorRoute(Graph* g, int *mirrorVertex, int *masterWorker, int mirrorSize);

#endif //DRONE_CUDA_GRAPH