#include "stdio.h"
extern "C" {
#include "graph.h"
#include "algorithm/common.h"
}

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

int addVertex(bool *exist, int *Global2Local, int *Local2Global, int localVertexSize, int u) {
    exist[u] = true;
    Global2Local[u] = localVertexSize;
    Local2Global[localVertexSize] = u;
    return localVertexSize + 1;
}

Graph* build_graph(int globalVertexSize, int edgeSize, int *u, int *v, int workerId, int workerNum, Comm *comm) {
    ncclUniqueId id;
    if (workerId == 0) {
        ncclGetUniqueId(&id);
        FILE *fp = fopen("./nccl.id", "w");
        fwrite(&id, sizeof(id), 1, fp);
        fclose(fp);
    }

    int GID = workerId % 4;
    Graph *g = (Graph *) malloc(sizeof(Graph));
    g -> GID = GID;
    cudaSetDevice(g -> GID);

    bool *exist;
    int *Global2Local, *Local2Global;
    int localVertexSize = 0;
    cudaMallocHost((void **) &exist, sizeof(bool) * globalVertexSize);
    cudaMallocHost((void **) &Global2Local, sizeof(int) * globalVertexSize);
    cudaMallocHost((void **) &Local2Global, sizeof(int) * globalVertexSize);

    cudaDeviceSynchronize();
    for (int i = 0; i < globalVertexSize; i++) {
        exist[i] = false;
        Global2Local[i] = -1;
    }

    for (int i = 0; i < edgeSize; i++) {
        if (!exist[u[i]]) localVertexSize = addVertex(exist, Global2Local, Local2Global, localVertexSize, u[i]);
        if (!exist[v[i]]) localVertexSize = addVertex(exist, Global2Local, Local2Global, localVertexSize, v[i]);
    }

    int *localOutDegree;
    cudaMallocHost((void **) &localOutDegree, sizeof(int) * localVertexSize);
    for (int i = 0; i < localVertexSize; i++) localOutDegree[i] = 0;
    for (int i = 0; i < edgeSize; i++) {
        localOutDegree[Global2Local[u[i]]]++;
    }

    int *addDiff;
    cudaMallocHost((void **) &addDiff, sizeof(int) * localVertexSize);
    for (int i = 0; i < localVertexSize; i++) addDiff[i] = 0;

    int *index, *dst;
    float *edgeLenHost;
    cudaMallocHost((void **) &index, sizeof(int) * (localVertexSize + 1));
    cudaMallocHost((void **) &dst, sizeof(int) * edgeSize);
    cudaMallocHost((void **) &edgeLenHost, sizeof(float) * edgeSize);
    index[0] = 0;
    for (int i = 1; i <= localVertexSize; i++) index[i] = index[i - 1] + localOutDegree[i - 1];
    for (int i = 0; i < edgeSize; i++) {
        int local_u = Global2Local[u[i]];
        int local_v = Global2Local[v[i]];
        dst[index[local_u] + addDiff[local_u]] = local_v;
        edgeLenHost[index[local_u] + addDiff[local_u]] = 1.0;
        addDiff[local_u]++;
    }

    cudaMalloc((void **) &(g->Global2Local), sizeof(int) * globalVertexSize);
    cudaMalloc((void **) &(g->Local2Global), sizeof(int) * localVertexSize);
    cudaMalloc((void **) &(g->index), sizeof(int) * (localVertexSize + 1));
    cudaMalloc((void **) &(g->dst), sizeof(int) * edgeSize);
    cudaMalloc((void **) &(g->edgeLen), sizeof(float) * edgeSize);

    cudaMallocManaged((void **) &(g->localVertexSize), sizeof(int));
    cudaMallocManaged((void **) &(g->globalVertexSize), sizeof(int));
    cudaMallocManaged((void **) &(g->edgeSize), sizeof(int));

    cudaMemcpy(g->Global2Local, Global2Local, sizeof(int) * globalVertexSize, cudaMemcpyHostToDevice);
    cudaMemcpy(g->Local2Global, Local2Global, sizeof(int) * localVertexSize, cudaMemcpyHostToDevice);
    cudaMemcpy(g->index, index, sizeof(int) * (localVertexSize + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(g->dst, dst, sizeof(int) * edgeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(g->edgeLen, edgeLenHost, sizeof(float) * edgeSize, cudaMemcpyHostToDevice);
    g->localVertexSize[0] = localVertexSize;
    g->globalVertexSize[0] = globalVertexSize;
    g->edgeSize[0] = edgeSize;

    cudaFreeHost(exist);
    cudaFreeHost(Global2Local);
    cudaFreeHost(Local2Global);
    cudaFreeHost(localOutDegree);
    cudaFreeHost(addDiff);
    cudaFreeHost(index);
    cudaFreeHost(dst);
    cudaFreeHost(edgeLenHost);

    g->blockSize = 256;
    g->gridSize = (*g->localVertexSize - 1) / g->blockSize + 1;
    if (g -> gridSize > 1024) {
        g -> gridSize = 1024;
    }

    if (workerId != 0) {
        FILE *fp = fopen("./nccl.id", "r");
        fread(&id, sizeof(id), 1, fp);
        fclose(fp);
    }

    NCCLCHECK(ncclCommInitRank(&comm->comm, workerNum, id, workerId));
    cudaStreamCreate(&comm->s);

    return g;
}

Graph* build_graph_withLen(int globalVertexSize, int edgeSize, int *u, int *v, float *edgeLen) {
    const size_t malloc_limit = size_t(1024) * size_t(1024) * size_t(1024) * 5;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, malloc_limit);

    Graph *g = (Graph *) malloc(sizeof(Graph));
    bool *exist;
    int *Global2Local, *Local2Global;
    int localVertexSize = 0;
    cudaMallocHost((void **) &exist, sizeof(bool) * globalVertexSize);
    cudaMallocHost((void **) &Global2Local, sizeof(int) * globalVertexSize);
    cudaMallocHost((void **) &Local2Global, sizeof(int) * globalVertexSize);

    cudaDeviceSynchronize();
    for (int i = 0; i < globalVertexSize; i++) {
        exist[i] = false;
        Global2Local[i] = -1;
    }

//    printf("cgo, edgeSize:%d\n", edgeSize);
//    printf("cgo, u[2994436]:%d, v[2994436]:%d\n", u[2994436], v[2994436]);

    for (int i = 0; i < edgeSize; i++) {
//        if (u[i] == 1386507 || v[i] == 1386507) {
//            printf("u[%d]:%d, v[%d]:%d\n", i, u[i], i, v[i]);
//        }
        if (!exist[u[i]]) localVertexSize = addVertex(exist, Global2Local, Local2Global, localVertexSize, u[i]);
        if (!exist[v[i]]) localVertexSize = addVertex(exist, Global2Local, Local2Global, localVertexSize, v[i]);
    }

//    printf("build: Global2Local[1386507]:%d\n", Global2Local[1386507]);

    int *localOutDegree;
    cudaMallocHost((void **) &localOutDegree, sizeof(int) * localVertexSize);
    for (int i = 0; i < localVertexSize; i++) localOutDegree[i] = 0;
    for (int i = 0; i < edgeSize; i++) {
        localOutDegree[Global2Local[u[i]]]++;
    }

    int *addDiff;
    cudaMallocHost((void **) &addDiff, sizeof(int) * localVertexSize);
    for (int i = 0; i < localVertexSize; i++) addDiff[i] = 0;

    int *index, *dst;
    float *edgeLenHost;
    cudaMallocHost((void **) &index, sizeof(int) * (localVertexSize + 1));
    cudaMallocHost((void **) &dst, sizeof(int) * edgeSize);
    cudaMallocHost((void **) &edgeLenHost, sizeof(float) * edgeSize);
    index[0] = 0;
    for (int i = 1; i <= localVertexSize; i++) index[i] = index[i - 1] + localOutDegree[i - 1];
    for (int i = 0; i < edgeSize; i++) {
        int local_u = Global2Local[u[i]];
        int local_v = Global2Local[v[i]];
        dst[index[local_u] + addDiff[local_u]] = local_v;
        edgeLenHost[index[local_u] + addDiff[local_u]] = edgeLen[i];
        addDiff[local_u]++;
    }

    cudaMalloc((void **) &(g->Global2Local), sizeof(int) * globalVertexSize);
    cudaMalloc((void **) &(g->Local2Global), sizeof(int) * localVertexSize);
    cudaMalloc((void **) &(g->index), sizeof(int) * (localVertexSize + 1));
    cudaMalloc((void **) &(g->dst), sizeof(int) * edgeSize);
    cudaMalloc((void **) &(g->edgeLen), sizeof(float) * edgeSize);

    cudaMallocManaged((void **) &(g->localVertexSize), sizeof(int));
    cudaMallocManaged((void **) &(g->globalVertexSize), sizeof(int));
    cudaMallocManaged((void **) &(g->edgeSize), sizeof(int));

    cudaMemcpy(g->Global2Local, Global2Local, sizeof(int) * globalVertexSize, cudaMemcpyHostToDevice);
    cudaMemcpy(g->Local2Global, Local2Global, sizeof(int) * localVertexSize, cudaMemcpyHostToDevice);
    cudaMemcpy(g->index, index, sizeof(int) * (localVertexSize + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(g->dst, dst, sizeof(int) * edgeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(g->edgeLen, edgeLenHost, sizeof(float) * edgeSize, cudaMemcpyHostToDevice);
    g->localVertexSize[0] = localVertexSize;
    g->globalVertexSize[0] = globalVertexSize;
    g->edgeSize[0] = edgeSize;

    cudaFreeHost(exist);
    cudaFreeHost(Global2Local);
    cudaFreeHost(Local2Global);
    cudaFreeHost(localOutDegree);
    cudaFreeHost(addDiff);
    cudaFreeHost(index);
    cudaFreeHost(dst);
    cudaFreeHost(edgeLenHost);

    g->blockSize = 256;
    g->gridSize = (*g->localVertexSize - 1) / g->blockSize + 1;
    if (g -> gridSize > 1024) {
        g -> gridSize = 1024;
    }
    return g;
}


__global__ void calLocalMirrorNumber(int *Global2Local, int *masterVertex, int *mirrorNumber, int *MasterWorkerIndex, int masterSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < masterSize) {
        int localVertexId = Global2Local[masterVertex[id]];
        MasterWorkerIndex[localVertexId + 1] = mirrorNumber[id];
        id += stride;
    }
}

__global__ void setMaster2Workers(int *Master2Workers, int *MasterWorkerIndex, int *masterVertex, int *mirrorNumberSum, int *mirrorWorkers, int *Global2Local, int masterSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    while (id < masterSize) {
        int localVertexId = Global2Local[masterVertex[id]];
//        if (id % 1000 == 0) {
//            printf("cuda -> id:%d, masterVertex[id]:%d, localVertexId:%d\n", id, masterVertex[id], localVertexId);
//        }
        for (int i = 0; MasterWorkerIndex[localVertexId] + i < MasterWorkerIndex[localVertexId + 1]; i++) {
//            if (id == masterSize - 1) {
//                printf("cuda -> i:\n", i);
//                printf("cuda -> mirrorNumberSum[id] + i:\n", mirrorNumberSum[id] + i);
//            }
            Master2Workers[MasterWorkerIndex[localVertexId] + i] = mirrorWorkers[mirrorNumberSum[id] + i];
        }
        if (MasterWorkerIndex[localVertexId + 1] - MasterWorkerIndex[localVertexId] != mirrorNumberSum[id + 1] - mirrorNumberSum[id]) {
            printf("Error, index error for id:%d\n", id);
        }
        id += stride;
    }
}

void addMasterRoute(Graph* g, int *masterVertex, int *mirrorNumber, int *mirrorWorkers, int masterSize, int mirrorWorkerSize) {
//    int device;
//    cudaGetDevice(&device);
//    printf("addMasterRoute: device: %d\n", device);
    cudaSetDevice(g -> GID);

    g -> MasterSize = masterSize;
    g -> MirrorWorkerSize = mirrorWorkerSize;

    int *MasterWorkerIndex;
    int localVertexSize = getLocalVertexSize(g);
    CHECK(cudaMallocHost((void **) &MasterWorkerIndex, sizeof(int) * (localVertexSize + 1)));
    CHECK(cudaMalloc((void **) &g->MasterWorkerIndex, sizeof(int) * (localVertexSize + 1)));

    int *masterVertexCUDA, *mirrorNumberCUDA, *mirrorWorkersCUDA;
    CHECK(cudaMalloc((void **) &masterVertexCUDA, sizeof(int) * masterSize));
    CHECK(cudaMalloc((void **) &mirrorNumberCUDA, sizeof(int) * masterSize));
    CHECK(cudaMalloc((void **) &mirrorWorkersCUDA, sizeof(int) * mirrorWorkerSize));
    cudaMemcpy(masterVertexCUDA, masterVertex, sizeof(int) * masterSize, cudaMemcpyHostToDevice);
    cudaMemcpy(mirrorNumberCUDA, mirrorNumber, sizeof(int) * masterSize, cudaMemcpyHostToDevice);
    cudaMemcpy(mirrorWorkersCUDA, mirrorWorkers, sizeof(int) * mirrorWorkerSize, cudaMemcpyHostToDevice);

    calLocalMirrorNumber<<<g->gridSize, g->blockSize>>>(g->Global2Local, masterVertexCUDA, mirrorNumberCUDA,
                                                        g->MasterWorkerIndex, masterSize);
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(MasterWorkerIndex, g->MasterWorkerIndex, sizeof(int) * (localVertexSize + 1), cudaMemcpyDeviceToHost));
    int *mirrorNumberSum;
    CHECK(cudaMallocManaged((void **) &mirrorNumberSum, sizeof(int) * (localVertexSize + 1)));
    for (int i = 1; i <= localVertexSize; i++) {
        MasterWorkerIndex[i] += MasterWorkerIndex[i - 1];
        mirrorNumberSum[i] = mirrorNumberSum[i - 1] + mirrorNumber[i - 1];
    }
    cudaMemcpy(g->MasterWorkerIndex, MasterWorkerIndex, sizeof(int) * (localVertexSize + 1), cudaMemcpyHostToDevice);

    CHECK(cudaMalloc((void **) &g->Master2Workers, sizeof(int) * mirrorWorkerSize));
//    printf("masterSize:%d\n", masterSize);
    setMaster2Workers<<<g->gridSize, g->blockSize>>>(g->Master2Workers, g->MasterWorkerIndex, masterVertexCUDA,
                      mirrorNumberSum, mirrorWorkersCUDA, g->Global2Local, masterSize);
    cudaDeviceSynchronize();

    CHECK(cudaFree(masterVertexCUDA));
    CHECK(cudaFree(mirrorNumberCUDA));
    cudaFree(mirrorWorkersCUDA);
    cudaFreeHost(MasterWorkerIndex);
    CHECK(cudaFree(mirrorNumberSum));
}

__global__ void setMirror2Worker(int *Mirror2Worker, int *mirrorVertex, int *masterWorker, int *Global2Local, int mirrorSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    while (id < mirrorSize) {
        int localVertexId = Global2Local[mirrorVertex[id]];
        Mirror2Worker[localVertexId] = masterWorker[id];
        id += stride;
    }
}

void addMirrorRoute(Graph* g, int *mirrorVertex, int *masterWorker, int mirrorSize) {
    cudaSetDevice(g -> GID);
    g -> MirrorSize = mirrorSize;

    int localVertexSize = getLocalVertexSize(g);
    CHECK(cudaMallocManaged((void **) &g->Mirror2Worker, sizeof(int) * localVertexSize));
    for (int i = 0; i < localVertexSize; i++) g->Mirror2Worker[i] = -1;

    int *mirrorVertexCUDA, *masterWorkerCUDA;
    CHECK(cudaMalloc(&mirrorVertexCUDA, sizeof(int) * mirrorSize));
    CHECK(cudaMalloc(&masterWorkerCUDA, sizeof(int) * mirrorSize));
    cudaMemcpy(mirrorVertexCUDA, mirrorVertex, sizeof(int) * mirrorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(masterWorkerCUDA, masterWorker, sizeof(int) * mirrorSize, cudaMemcpyHostToDevice);

    setMirror2Worker<<<g->gridSize, g->blockSize>>>(g->Mirror2Worker, mirrorVertexCUDA, masterWorkerCUDA,
                                                    g->Global2Local, mirrorSize);
    cudaDeviceSynchronize();

    cudaFree(mirrorVertexCUDA);
    cudaFree(masterWorkerCUDA);
}

int getLocalVertexSize(Graph* g) {
    cudaSetDevice(g -> GID);
//    int localVertexSize;
//    cudaMemcpy(&localVertexSize, g -> localVertexSize, sizeof(int), cudaMemcpyDeviceToHost);
//    return localVertexSize;
    return *g->localVertexSize;
}

int getGlobalVertexSize(Graph* g) {
    cudaSetDevice(g -> GID);
    return *g->globalVertexSize;
}