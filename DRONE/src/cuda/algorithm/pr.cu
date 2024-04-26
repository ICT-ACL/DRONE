#include "stdio.h"
#include "cuda_runtime.h"
#include "nccl.h"
#include <iostream>
#include "time.h"
#include "math.h"
extern "C" {
#include "pr.h"
}

const float eps = 1e-6;
const float alpha = 0.85;

__global__ void pr_init(float *prValue, float *accValue, float *diffValue, float *globalDiff, int *outDegree,
                        int *active, int *index, int *dst, int *Mirror2Worker,  int localVertexSize, int globalVertexSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        accValue[id] = 0.0;
        prValue[id] = 1.0;

        if (outDegree[id] == 0) {
            if (Mirror2Worker[id] == -1) {
                atomicAdd(globalDiff, prValue[id] / globalVertexSize);
            }
        } else {
            float temp = prValue[id] / outDegree[id];
            for (int i = index[id]; i < index[id + 1]; i++) {
                int v = dst[i];
                atomicAdd(diffValue + v, temp);
                active[v] = 1;
            }
        }
        id += stride;
    }
}

__device__ float fatomicMax(float *addr, float value) {
    float ret = *addr;
    float old = ret, assumed;
    if (old >= value) return old;
    do {
        assumed = old;
        old = atomicCAS((unsigned int *) addr, __float_as_int(assumed), __float_as_int(value));

    } while (old != assumed);
    return ret;
}

__global__ void pr_increment(float *prValue, float *accValue, float *diffValue, float *globalDiff, float *globalAcc, int *outDegree,
                             int *active, int *index, int *dst, int *Mirror2Worker,  int localVertexSize, int globalVertexSize, float *maxDiff) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        float pr = alpha * (accValue[id] + *globalAcc) + (1 - alpha);
        float diff = pr - prValue[id];
        if (diff < 0) diff = -diff;

        if (diff < eps) {
            id += stride;
            continue;
        }

        fatomicMax(maxDiff, diff);

        if (outDegree[id] == 0) {
            if (Mirror2Worker[id] == -1) {
                atomicAdd(globalDiff, (pr - prValue[id]) / globalVertexSize);
            }
        } else {
            float temp = (pr - prValue[id]) / outDegree[id];
            for (int i = index[id]; i < index[id + 1]; i++) {
                int v = dst[i];
                atomicAdd(diffValue + v, temp);
                active[v] = 1;
            }
        }
//        if (id == 1024) {
//            printf("id: %d, prValue:%f, globalDiff:%f\n", id, prValue[id], *globalDiff);
//        }
        prValue[id] = pr;

        id += stride;
    }
}

__global__ void pr_cal_mirror_sendLen(int *sendLen, int *Mirror2Worker, int *active, int localVertexSize ) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        if (active[id] == 0 || Mirror2Worker[id] == -1) {
            id += stride;
            continue;
        }
        int *address = sendLen + Mirror2Worker[id];
        atomicAdd(address, 1);

        id += stride;
    }
}

__global__ void pr_cal_master_sendLen(int *sendLen, int *MasterWorkerIndex, int *Master2Workers, int *active, int *updateByMessage, int localVertexSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        if (active[id] == 0 && updateByMessage[id] == 0) {
            id += stride;
            continue;
        }
        for (int i = MasterWorkerIndex[id]; i < MasterWorkerIndex[id + 1]; i++) {
            int worker = Master2Workers[i];
            int *address = sendLen + worker;
            atomicAdd(address, 1);
        }

        id += stride;
    }
}

__global__ void pr_generate_sendBuff_mirror(float *sendDiffValue, int *sendID, int *sendIndex, int *sendDiff,
                                            int *Mirror2Worker, int *Local2Global, float *diffValue, int *active,
                                            int localVertexSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        if (active[id] == 0 || Mirror2Worker[id] == -1) {
            id += stride;
            continue;
        }
        int globalId = Local2Global[id];
        float val = diffValue[id];
        int worker = Mirror2Worker[id];
        int *address = sendDiff + worker;
        int diff = atomicAdd(address, 1);

        sendID[sendIndex[worker] + diff] = globalId;
        sendDiffValue[sendIndex[worker] + diff] = val;

        id += stride;
    }
}

__global__ void pr_generate_sendBuff_master(float *sendDiffValue, int *sendID, int *sendIndex, int *sendDiff,
                                         int *MasterWorkerIndex, int *Master2Workers, int *Local2Global, float *diffValue,
                                         int *active, int *updateByMessage, int localVertexSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        if (active[id] == 0 && updateByMessage[id] == 0) {
            id += stride;
            continue;
        }
        int globalId = Local2Global[id];
        float val = diffValue[id];

        for (int i = MasterWorkerIndex[id]; i < MasterWorkerIndex[id + 1]; i++) {
            int worker = Master2Workers[i];
            int *address = sendDiff + worker;
            int diff = atomicAdd(address, 1);
            sendID[sendIndex[worker] + diff] = globalId;
            sendDiffValue[sendIndex[worker] + diff] = val;
        }
        id += stride;
    }
}

__global__ void pr_process_recvBuff_Add(float *recvDiff, int *recvID, int recvLen, float *diffValue, int *Global2Local, int *updateByMessage) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < recvLen) {
        int localID = Global2Local[recvID[id]];
        float val = recvDiff[id];

        if (localID == -1) {
            printf("error global id :%d!\n", recvID[id]);
            id += stride;
            continue;
        }

        atomicAdd(&diffValue[localID], val);
        updateByMessage[localID] = 1;

        id += stride;
    }
}

__global__ void pr_process_recvBuff_Exch(float *recvDiff, int *recvID, int recvLen, float *diffValue, int *Global2Local, int *updateByMessage) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < recvLen) {
        int localID = Global2Local[recvID[id]];
        float val = recvDiff[id];

        if (localID == -1) {
            printf("error global id :%d!\n", recvID[id]);
            id += stride;
            continue;
        }

        atomicExch(&diffValue[localID], val);
        updateByMessage[localID] = 1;

        id += stride;
    }
}

void Mirror2MasterSend(PRValues *values, Graph *graph, Comm *comm, int workerId, int workerNum, Response* res) {
    int *sendLen, *recvLen;
    CUDACHECK(cudaMallocManaged(&sendLen, sizeof(int) * workerNum));
    cudaMallocManaged(&recvLen, sizeof(int) * workerNum);
    for (int i = 0; i < workerNum; i++) {
        sendLen[i] = 0;
    }
    int localVertexSize = getLocalVertexSize(graph);

    pr_cal_mirror_sendLen<<<graph->gridSize, graph->blockSize>>>(sendLen, graph->Mirror2Worker, graph->active, localVertexSize);

    CUDACHECK(cudaDeviceSynchronize());
//    for (int i = 0; i < workerNum; i++) {
//        printf("send %d -> %d: %d\n", workerId, i, sendLen[i]);
//    }

    ncclGroupStart();
    for (int i = 0; i < workerNum; i++) {
        if (i == workerId) continue;
        NCCLCHECK(ncclSend(sendLen + i, 1, ncclInt, i, comm->comm, comm->s));
        NCCLCHECK(ncclRecv(recvLen + i, 1, ncclInt, i, comm->comm, comm->s));
    }
    ncclGroupEnd();
    CUDACHECK(cudaStreamSynchronize(comm->s));

//    for (int i = 0; i < workerNum; i++) {
//        printf("recv %d -> %d: %d\n", i, workerId, recvLen[i]);
//    }
//    printf("--------------------------\n");

    int *sendIndex, *sendDiff, *recvIndex;
    cudaMallocManaged(&sendIndex, sizeof(int) * (workerNum + 1));
    cudaMalloc(&sendDiff, sizeof(int) * workerNum);
    CUDACHECK(cudaMallocHost(&recvIndex, sizeof(int) * (workerNum + 1)));
    recvIndex[0] = 0;
    sendIndex[0] = 0;
    CUDA_FILL_INT<<<graph->gridSize, graph->blockSize>>>(sendDiff, 0, workerNum);

    for (int i = 1; i <= workerNum; i++) {
        sendIndex[i] = sendIndex[i - 1] + sendLen[i - 1];
        recvIndex[i] = recvIndex[i - 1] + recvLen[i - 1];
    }

    pr_generate_sendBuff_mirror<<<graph->gridSize, graph->blockSize>>>(values->sendDiff, values->sendID, sendIndex, sendDiff,
                                graph->Mirror2Worker, graph->Local2Global, values->diffValue, graph->active, localVertexSize);

    CUDACHECK(cudaDeviceSynchronize());

    ncclGroupStart();
    for (int i = 0; i < workerNum; i++) {
        if (i == workerId) continue;
        ncclSend(values->sendID + sendIndex[i], sendLen[i], ncclInt, i, comm->comm, comm->s);
        ncclRecv(values->recvID + recvIndex[i], recvLen[i], ncclInt, i, comm->comm, comm->s);
    }
    ncclGroupEnd();
    cudaStreamSynchronize(comm->s);

    ncclGroupStart();
    for (int i = 0; i < workerNum; i++) {
        if (i == workerId) continue;
        ncclSend(values->sendDiff + sendIndex[i], sendLen[i], ncclFloat, i, comm->comm, comm->s);
        ncclRecv(values->recvDiff + recvIndex[i], recvLen[i], ncclFloat, i, comm->comm, comm->s);
    }
    ncclGroupEnd();
    cudaStreamSynchronize(comm->s);

    int sendSize = sendIndex[workerNum];
    int recvSize = recvIndex[workerNum];

    res->Mirror2MasterSendSize = sendSize;
    res->Mirror2MasterRecvSize = recvSize;

//    process_recvBuff<<<graph->gridSize, graph->blockSize>>>(values, graph, recvSize, atomicAdd);
    pr_process_recvBuff_Add<<<graph->gridSize, graph->blockSize>>>(values->recvDiff, values->recvID, recvSize, values->diffValue,
                                                               graph->Global2Local, graph->updateByMessage);
    CUDACHECK(cudaDeviceSynchronize());

    cudaFree(sendLen);
    cudaFree(recvLen);
    cudaFree(sendIndex);
    cudaFree(sendDiff);
    cudaFreeHost(recvIndex);
}

void Master2MirrorSend(PRValues *values, Graph *graph, Comm *comm, int workerId, int workerNum, Response* res) {
    int *sendLen, *recvLen;
    cudaMallocManaged(&sendLen, sizeof(int) * workerNum);
    cudaMallocManaged(&recvLen, sizeof(int) * workerNum);
    for (int i = 0; i < workerNum; i++) {
        sendLen[i] = 0;
    }
    int localVertexSize = getLocalVertexSize(graph);
    pr_cal_master_sendLen<<<graph->gridSize, graph->blockSize>>>(sendLen, graph->MasterWorkerIndex, graph->Master2Workers, graph->active, graph->updateByMessage, localVertexSize);
    CUDACHECK(cudaDeviceSynchronize());

    ncclGroupStart();
    for (int i = 0; i < workerNum; i++) {
        if (i == workerId) continue;
        ncclSend(sendLen + i, 1, ncclInt, i, comm->comm, comm->s);
        ncclRecv(recvLen + i, 1, ncclInt, i, comm->comm, comm->s);
    }
    ncclGroupEnd();
    CUDACHECK(cudaStreamSynchronize(comm->s));

    int *sendIndex, *sendDiff, *recvIndex;
    cudaMallocManaged(&sendIndex, sizeof(int) * (workerNum + 1));
    cudaMalloc(&sendDiff, sizeof(int) * workerNum);
    cudaMallocHost(&recvIndex, sizeof(int) * (workerNum + 1));
    recvIndex[0] = 0;
    sendIndex[0] = 0;
    CUDA_FILL_INT<<<graph->gridSize, graph->blockSize>>>(sendDiff, 0, workerNum);

    for (int i = 1; i <= workerNum; i++) {
        sendIndex[i] = sendIndex[i - 1] + sendLen[i - 1];
        recvIndex[i] = recvIndex[i - 1] + recvLen[i - 1];
    }

    pr_generate_sendBuff_master<<<graph->gridSize, graph->blockSize>>>(values->sendDiff, values->sendID, sendIndex, sendDiff,
                                graph->MasterWorkerIndex, graph->Master2Workers, graph->Local2Global, values->diffValue,
                                graph->active, graph->updateByMessage, localVertexSize);
    CUDACHECK(cudaDeviceSynchronize());

    ncclGroupStart();
    for (int i = 0; i < workerNum; i++) {
        if (i == workerId) continue;
        ncclSend(values->sendID + sendIndex[i], sendLen[i], ncclInt, i, comm->comm, comm->s);
        ncclRecv(values->recvID + recvIndex[i], recvLen[i], ncclInt, i, comm->comm, comm->s);
    }
    ncclGroupEnd();
    CUDACHECK(cudaStreamSynchronize(comm->s));

    ncclGroupStart();
    for (int i = 0; i < workerNum; i++) {
        if (i == workerId) continue;
        ncclSend(values->sendDiff + sendIndex[i], sendLen[i], ncclFloat, i, comm->comm, comm->s);
        ncclRecv(values->recvDiff + recvIndex[i], recvLen[i], ncclFloat, i, comm->comm, comm->s);
    }
    ncclGroupEnd();
    CUDACHECK(cudaStreamSynchronize(comm->s));

    int sendSize = sendIndex[workerNum];
    int recvSize = recvIndex[workerNum];

    res->Master2MirrorSendSize = sendSize;
    res->Master2MirrorRecvSize = recvSize;

    pr_process_recvBuff_Exch<<<graph->gridSize, graph->blockSize>>>(values->recvDiff, values->recvID, recvSize, values->diffValue,
                                                               graph->Global2Local, graph->updateByMessage);
    CUDACHECK(cudaDeviceSynchronize());

    NCCLCHECK(ncclAllReduce(values->globalDiff, values->globalDiff, 1, ncclFloat, ncclSum, comm->comm, comm->s));
    CUDACHECK(cudaDeviceSynchronize());

    cudaFree(sendLen);
    cudaFree(recvLen);
    cudaFree(sendIndex);
    cudaFree(sendDiff);
    cudaFreeHost(recvIndex);
}

Response PR_PEVal(Graph *g, PRValues* values, int workerId, int workerNum, Comm *comm) {
    clock_t start, end;
    start = clock();

    cudaSetDevice(g->GID);
    Response res;

    int localVertexSize = getLocalVertexSize(g);
    int globalVertexSize = getGlobalVertexSize(g);
    cudaMalloc(&g->active, sizeof(int) * localVertexSize);
    cudaMalloc(&values->prValue, sizeof(float) * localVertexSize);
    cudaMalloc(&values->accValue, sizeof(float) * localVertexSize);
    cudaMalloc(&values->diffValue, sizeof(float) * localVertexSize);
    cudaMalloc(&g->updateByMessage, sizeof(int) * localVertexSize);
    cudaMalloc(&values->globalDiff, sizeof(float));
    cudaMalloc(&values->globalAcc, sizeof(float));


    CUDA_FILL_float<<<g->gridSize, g->blockSize>>>(values->diffValue, 0.0f, localVertexSize);
    CUDA_FILL_float<<<1,32>>>(values->globalDiff, 0.0f, 1);
    CUDA_FILL_float<<<1,32>>>(values->globalAcc, 0.0f, 1);
    CUDACHECK(cudaDeviceSynchronize());

    pr_init<<<g->gridSize, g->blockSize>>>(values->prValue, values->accValue, values->diffValue, values->globalDiff, values->outDegree,
            g->active, g->index, g->dst, g->Mirror2Worker, localVertexSize, globalVertexSize);
    CUDACHECK(cudaDeviceSynchronize());

    res.LocalVertexSize = localVertexSize;

    cudaMalloc(&values->sendID, sizeof(int) * max(g->MirrorSize, g->MirrorWorkerSize));
    cudaMalloc(&values->sendDiff, sizeof(float) * max(g->MirrorSize, g->MirrorWorkerSize));
    cudaMalloc(&values->recvID, sizeof(int) * max(g->MirrorSize, g->MirrorWorkerSize));
    cudaMalloc(&values->recvDiff, sizeof(float) * max(g->MirrorSize, g->MirrorWorkerSize));

    printf("Buffer apply size:%d\n", max(g->MirrorSize, g->MirrorWorkerSize));
    printf("MirrorSize:%d, MasterSize:%d, MirrorWorkerSize:%d\n", g->MirrorSize, g->MasterSize, g->MirrorWorkerSize);

    end = clock();
    res.CalTime = (float) (end - start) / CLOCKS_PER_SEC;

    start = clock();
    printf("Mirror2MasterSend start!\n");
    Mirror2MasterSend(values, g, comm, workerId, workerNum, &res);
    printf("Mirror2MasterSend ok!\n");
    Master2MirrorSend(values, g, comm, workerId, workerNum, &res);
    printf("Master2MirrorSend ok!\n");
    end = clock();
    res.SendTime = (float) (end - start) / CLOCKS_PER_SEC;

    printf("countActive start!\n");
    res.VisitedSize = countActive(g->active, localVertexSize, g->gridSize, g->blockSize);
    printf("countActive ok!\n");
    return res;
}

__global__ void pr_process_messages(float *diffValue, float *accValue,  float* globalDiff, float *globalAcc,
                                    int *active, int *updateByMessage, int maxLen) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    if (id == 0) {
        *globalAcc += *globalDiff;
        *globalDiff = 0;
    }

    while (id < maxLen) {
        active[id] = 0;
        updateByMessage[id] = 0;
        accValue[id] += diffValue[id];
        diffValue[id] = 0;

        id += stride;
    }
}

Response PR_IncEVal(Graph *g, PRValues* values, int workerId, int workerNum, Comm *comm) {
    clock_t start, end;
    start = clock();

    cudaSetDevice(g->GID);
    Response res;

    int localVertexSize = getLocalVertexSize(g);
    int globalVertexSize = getGlobalVertexSize(g);
    res.LocalVertexSize = localVertexSize;

    pr_process_messages<<<g->gridSize, g->blockSize>>>(values->diffValue, values->accValue,  values->globalDiff, values->globalAcc,
                        g->active, g->updateByMessage, localVertexSize);
    CUDACHECK(cudaDeviceSynchronize());

    float *maxDiff;
    cudaMallocManaged(&maxDiff, sizeof(float));
    *maxDiff = -1;

//        printf("before sssp_exec\n");
    pr_increment<<<g->gridSize, g->blockSize>>>(values->prValue, values->accValue, values->diffValue, values->globalDiff, values->globalAcc, values->outDegree,
                 g->active, g->index, g->dst, g->Mirror2Worker, localVertexSize, globalVertexSize, maxDiff);
    CUDACHECK(cudaDeviceSynchronize());
//        printf("Inc --- step:%d\n", step);
    printf("MaxDiff:%f\n", *maxDiff);
    cudaFree(maxDiff);

    end = clock();
    res.CalTime = (float) (end - start) / CLOCKS_PER_SEC;

    start = clock();
    Mirror2MasterSend(values, g, comm, workerId, workerNum, &res);
    Master2MirrorSend(values, g, comm, workerId, workerNum, &res);
    end = clock();
    res.SendTime = (float) (end - start) / CLOCKS_PER_SEC;

    res.VisitedSize = countActive(g->active, localVertexSize, g->gridSize, g->blockSize);
    return res;
}

__global__ void setOutDegree(int *localOutDegree, int *globalOutDegree, int *Local2Global, int localVertexSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        int globalID = Local2Global[id];
        localOutDegree[id] = globalOutDegree[globalID];

        id += stride;
    }
}

void loadOutDegree(Graph *g, PRValues* values, int *globalOutDegree) {
    cudaSetDevice(g -> GID);

    int globalVertexSize = *g->globalVertexSize;
    int localVertexSize = *g->localVertexSize;
    int *globalOutDegreeDevice;
    cudaMalloc(&values->outDegree, sizeof(int) * localVertexSize);
    CUDACHECK(cudaMalloc(&globalOutDegreeDevice, sizeof(int) * globalVertexSize));
    cudaMemcpy(globalOutDegreeDevice, globalOutDegree, sizeof(int) * globalVertexSize, cudaMemcpyHostToDevice);

    setOutDegree<<<g->gridSize, g->blockSize>>>(values->outDegree, globalOutDegreeDevice, g->Local2Global, localVertexSize);

    cudaFree(globalOutDegreeDevice);
}
