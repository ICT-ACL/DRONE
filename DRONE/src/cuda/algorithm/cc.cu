#include "stdio.h"
#include "cuda_runtime.h"
#include "nccl.h"
#include <iostream>
#include "time.h"
extern "C" {
#include "cc.h"
}

__global__ void cc_init(int *active, int *ccValue, int localVertexSize, bool *isUpdate) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        active[id] = 0;
        ccValue[id] = id;
        *isUpdate = true;
        id += stride;
    }
}

__global__ void cc_exec(int *active, int *ccValue, int *index, int *dst, int localVertexSize, int step, bool *isUpdate) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        if (active[id] != step) {
            id += stride;
            continue;
        }

        int nowCCValue = ccValue[id];
        for (int i = index[id]; i < index[id + 1]; i++) {
            int v = dst[i];

            int old = atomicMin(&ccValue[v], nowCCValue);
            if (old > nowCCValue) {
                active[v] = step + 1;
//                if (step == 1) printf("active: %d, old:%f, newDis:%f\n", v, old, newDis);
                *isUpdate = true;
            }
//            if (step == 1) printf("new search vertex:%d, newDis:%f, old:%f, nowDis:%f, edgeLen:%f\n", v, newDis, old, nowDis, edgeLen[i]);
        }
        id += stride;
    }
}

__global__ void cc_cal_mirror_sendLen(int *sendLen, int *Mirror2Worker, int *active, int localVertexSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        if (active[id] <= 0 || Mirror2Worker[id] == -1) {
            id += stride;
            continue;
        }
        int *address = sendLen + Mirror2Worker[id];
        atomicAdd(address, 1);

        id += stride;
    }
}

__global__ void cc_cal_master_sendLen(int *sendLen, int *MasterWorkerIndex, int *Master2Workers, int *active, int *updateByMessage,
                   int localVertexSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        if (active[id] <= 0 && updateByMessage[id] == 0) {
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

__global__ void cc_generate_sendBuff_mirror(int *sendBuffCCValue, int *sendBuffId, int *sendIndex, int *sendDiff,
                                         int *Mirror2Worker, int *Local2Global, int *ccValue, int *active,
                                         int localVertexSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        if (active[id] <= 0 || Mirror2Worker[id] == -1) {
            id += stride;
            continue;
        }
        int globalId = Local2Global[id];
        int value = ccValue[id];
        int worker = Mirror2Worker[id];
        int *address = sendDiff + worker;
        int diff = atomicAdd(address, 1);

        sendBuffId[sendIndex[worker] + diff] = globalId;
        sendBuffCCValue[sendIndex[worker] + diff] = value;

        id += stride;
    }
}

__global__ void cc_generate_sendBuff_master(int *sendBuffCCValue, int *sendBuffId, int *sendIndex, int *sendDiff,
                                         int *MasterWorkerIndex, int *Master2Workers, int *Local2Global,
                                         int *ccValue,
                                         int *active, int *updateByMessage, int localVertexSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        if (active[id] <= 0 && updateByMessage[id] == 0) {
            id += stride;
            continue;
        }
        int globalId = Local2Global[id];
        int value = ccValue[id];

        for (int i = MasterWorkerIndex[id]; i < MasterWorkerIndex[id + 1]; i++) {
            int worker = Master2Workers[i];
            int *address = sendDiff + worker;
            int diff = atomicAdd(address, 1);
            sendBuffId[sendIndex[worker] + diff] = globalId;
            sendBuffCCValue[sendIndex[worker] + diff] = value;
        }

        id += stride;
    }
}

__global__ void cc_process_recvBuff(int *recvBuffCCValue, int *recvBuffId, int recvLen, int *Global2Local, int *ccValue,
                 int *updateByMessage) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < recvLen) {
        int localId = Global2Local[recvBuffId[id]];
        int value = recvBuffCCValue[id];

        if (localId == -1) {
            printf("error global id :%d!\n", recvBuffId[id]);
            id += stride;
            continue;
        }

        int old = atomicMin(&ccValue[localId], value);
        if (old > value) {
            updateByMessage[localId] = 1;
        }

        id += stride;
    }
}


void cc_Mirror2MasterSend(CCValues *values, Graph *g, ncclComm_t *comm, cudaStream_t *s, int workerId, int workerNum,
                       int localVertexSize, Response *res) {
    int *sendLen, *recvLen;
    CUDACHECK(cudaMallocManaged(&sendLen, sizeof(int) * workerNum));
    cudaMallocManaged(&recvLen, sizeof(int) * workerNum);
    for (int i = 0; i < workerNum; i++) {
        sendLen[i] = 0;
    }

    cc_cal_mirror_sendLen<<<g->gridSize, g->blockSize>>>(sendLen, g->Mirror2Worker, g->active, localVertexSize);

    CUDACHECK(cudaDeviceSynchronize());
//    for (int i = 0; i < workerNum; i++) {
//        printf("send %d -> %d: %d\n", workerId, i, sendLen[i]);
//    }

    ncclGroupStart();
    for (int i = 0; i < workerNum; i++) {
        if (i == workerId) continue;
        NCCLCHECK(ncclSend(sendLen + i, 1, ncclInt, i, *comm, *s));
        NCCLCHECK(ncclRecv(recvLen + i, 1, ncclInt, i, *comm, *s));
    }
    ncclGroupEnd();
    CUDACHECK(cudaStreamSynchronize(*s));

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
    CUDA_FILL_INT<<<g->gridSize, g->blockSize>>>(sendDiff, 0, workerNum);

    for (int i = 1; i <= workerNum; i++) {
        sendIndex[i] = sendIndex[i - 1] + sendLen[i - 1];
        recvIndex[i] = recvIndex[i - 1] + recvLen[i - 1];
    }

    cc_generate_sendBuff_mirror<<<g->gridSize, g->blockSize>>>(values->sendCCValue, values->sendID, sendIndex,
                                                            sendDiff,
                                                            g->Mirror2Worker, g->Local2Global, values->ccValue,
                                                            g->active, localVertexSize);

    CUDACHECK(cudaDeviceSynchronize());

    ncclGroupStart();
    for (int i = 0; i < workerNum; i++) {
        if (i == workerId) continue;
        ncclSend(values->sendID + sendIndex[i], sendLen[i], ncclInt, i, *comm, *s);
        ncclRecv(values->recvID + recvIndex[i], recvLen[i], ncclInt, i, *comm, *s);
    }
    ncclGroupEnd();
    cudaStreamSynchronize(*s);

    ncclGroupStart();
    for (int i = 0; i < workerNum; i++) {
        if (i == workerId) continue;
        ncclSend(values->sendCCValue + sendIndex[i], sendLen[i], ncclFloat, i, *comm, *s);
        ncclRecv(values->recvCCValue + recvIndex[i], recvLen[i], ncclFloat, i, *comm, *s);
    }
    ncclGroupEnd();
    cudaStreamSynchronize(*s);

    int sendSize = sendIndex[workerNum];
    int recvSize = recvIndex[workerNum];

    res->Mirror2MasterSendSize = sendSize;
    res->Mirror2MasterRecvSize = recvSize;

    cc_process_recvBuff<<<g->gridSize, g->blockSize>>>(values->recvCCValue, values->recvID, recvSize, g->Global2Local,
                                                    values->ccValue, g->updateByMessage);
    CUDACHECK(cudaDeviceSynchronize());

    cudaFree(sendLen);
    cudaFree(recvLen);
    cudaFree(sendIndex);
    cudaFree(sendDiff);
    cudaFreeHost(recvIndex);
}

void cc_Master2MirrorSend(CCValues *values, Graph *g, ncclComm_t *comm, cudaStream_t *s, int workerId, int workerNum,
                       int localVertexSize, Response *res) {
    int *sendLen, *recvLen;
    cudaMallocManaged(&sendLen, sizeof(int) * workerNum);
    cudaMallocManaged(&recvLen, sizeof(int) * workerNum);
    for (int i = 0; i < workerNum; i++) {
        sendLen[i] = 0;
    }
    cc_cal_master_sendLen<<<g->gridSize, g->blockSize>>>(sendLen, g->MasterWorkerIndex, g->Master2Workers, g->active,
                                                      g->updateByMessage, localVertexSize);
    CUDACHECK(cudaDeviceSynchronize());

//    for (int i = 0; i < workerNum; i++) {
//        printf("send %d -> %d: %d\n", workerId, i, sendLen[i]);
//    }

    ncclGroupStart();
    for (int i = 0; i < workerNum; i++) {
        if (i == workerId) continue;
        ncclSend(sendLen + i, 1, ncclInt, i, *comm, *s);
        ncclRecv(recvLen + i, 1, ncclInt, i, *comm, *s);
    }
    ncclGroupEnd();
    CUDACHECK(cudaStreamSynchronize(*s));

//    for (int i = 0; i < workerNum; i++) {
//        printf("recv %d -> %d: %d\n", i, workerId, recvLen[i]);
//    }

    int *sendIndex, *sendDiff, *recvIndex;
    cudaMallocManaged(&sendIndex, sizeof(int) * (workerNum + 1));
    cudaMalloc(&sendDiff, sizeof(int) * workerNum);
    cudaMallocHost(&recvIndex, sizeof(int) * (workerNum + 1));
    recvIndex[0] = 0;
    sendIndex[0] = 0;
    CUDA_FILL_INT<<<g->gridSize, g->blockSize>>>(sendDiff, 0, workerNum);

    for (int i = 1; i <= workerNum; i++) {
        sendIndex[i] = sendIndex[i - 1] + sendLen[i - 1];
        recvIndex[i] = recvIndex[i - 1] + recvLen[i - 1];
    }

    cc_generate_sendBuff_master<<<g->gridSize, g->blockSize>>>(values->sendCCValue, values->sendID, sendIndex,
                                                            sendDiff,
                                                            g->MasterWorkerIndex, g->Master2Workers,
                                                            g->Local2Global,
                                                            values->ccValue,
                                                            g->active, g->updateByMessage, localVertexSize);
    CUDACHECK(cudaDeviceSynchronize());

    ncclGroupStart();
    for (int i = 0; i < workerNum; i++) {
        if (i == workerId) continue;
        ncclSend(values->sendID + sendIndex[i], sendLen[i], ncclInt, i, *comm, *s);
        ncclRecv(values->recvID + recvIndex[i], recvLen[i], ncclInt, i, *comm, *s);
    }
    ncclGroupEnd();
    CUDACHECK(cudaStreamSynchronize(*s));

    ncclGroupStart();
    for (int i = 0; i < workerNum; i++) {
        if (i == workerId) continue;
        ncclSend(values->sendCCValue + sendIndex[i], sendLen[i], ncclFloat, i, *comm, *s);
        ncclRecv(values->recvCCValue + recvIndex[i], recvLen[i], ncclFloat, i, *comm, *s);
    }
    ncclGroupEnd();
    CUDACHECK(cudaStreamSynchronize(*s));

    int sendSize = sendIndex[workerNum];
    int recvSize = recvIndex[workerNum];

    res->Master2MirrorSendSize = sendSize;
    res->Master2MirrorRecvSize = recvSize;

    cc_process_recvBuff<<<g->gridSize, g->blockSize>>>(values->recvCCValue, values->recvID, recvSize, g->Global2Local,
                                                    values->ccValue, g->updateByMessage);
    CUDACHECK(cudaDeviceSynchronize());

    cudaFree(sendLen);
    cudaFree(recvLen);
    cudaFree(sendIndex);
    cudaFree(sendDiff);
    cudaFreeHost(recvIndex);
}

Response CC_PEVal(Graph *g, CCValues *values, int workerId, int workerNum, Comm *comm) {
    clock_t start, end;
    start = clock();

    cudaSetDevice(g->GID);
    printf("C call CC_PEVal!\n");
    Response res;

    int localVertexSize = getLocalVertexSize(g);
    cudaMalloc(&g->active, sizeof(int) * localVertexSize);
    cudaMalloc(&values->ccValue, sizeof(int) * localVertexSize);
    cudaMalloc(&g->updateByMessage, sizeof(int) * localVertexSize);

    bool *isUpdate;
    cudaMallocManaged(&isUpdate, sizeof(bool));
    *isUpdate = false;

    cc_init<<<g->gridSize, g->blockSize>>>(g->active, values->ccValue, localVertexSize, isUpdate);
    cudaDeviceSynchronize();

    res.LocalVertexSize = localVertexSize;

    int step = 0;
    while (*isUpdate) {
        *isUpdate = false;
        cc_exec<<<g->gridSize, g->blockSize>>>(g->active, values->ccValue, g->index, g->dst, localVertexSize,
                                               step, isUpdate);
        cudaDeviceSynchronize();
        step++;
    }

    cudaMalloc(&values->sendID, sizeof(int) * max(g->MirrorSize, g->MirrorWorkerSize));
    cudaMalloc(&values->sendCCValue, sizeof(int) * max(g->MirrorSize, g->MirrorWorkerSize));
    cudaMalloc(&values->recvID, sizeof(int) * max(g->MirrorSize, g->MirrorWorkerSize));
    cudaMalloc(&values->recvCCValue, sizeof(int) * max(g->MirrorSize, g->MirrorWorkerSize));

//    printf("Buffer apply size:%d\n", max(g->MirrorSize, g->MirrorWorkerSize));
//    printf("MirrorSize:%d, MasterSize:%d, MirrorWorkerSize:%d\n", g->MirrorSize, g->MasterSize, g->MirrorWorkerSize);

    end = clock();
    res.CalTime = (float) (end - start) / CLOCKS_PER_SEC;

    start = clock();
    printf("cc_Mirror2MasterSend start!\n");
    cc_Mirror2MasterSend(values, g, &comm->comm, &comm->s, workerId, workerNum, localVertexSize, &res);
    printf("cc_Mirror2MasterSend ok!\n");
    cc_Master2MirrorSend(values, g, &comm->comm, &comm->s, workerId, workerNum, localVertexSize, &res);
    printf("cc_Master2MirrorSend ok!\n");
    end = clock();
    res.SendTime = (float) (end - start) / CLOCKS_PER_SEC;


    cudaFree(isUpdate);
    printf("countActive start!\n");
    res.VisitedSize = countActive(g->active, localVertexSize, g->gridSize, g->blockSize);
    printf("countActive ok!\n");
    return res;
}

__global__ void cc_process_messages(int *active, int *updateByMessage, int maxLen, bool *isUpdate) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < maxLen) {
        if (updateByMessage[id] != 0) {
            active[id] = 0;
            updateByMessage[id] = 0;
            *isUpdate = true;
        } else {
            active[id] = -1;
        }
        id += stride;
    }
}

Response CC_IncEVal(Graph *g, CCValues *values, int workerId, int workerNum, Comm *comm) {
    clock_t start, end;
    start = clock();

    cudaSetDevice(g->GID);
//    printf("C call SSSP_IncEVal!\n");

    Response res;

    int localVertexSize = getLocalVertexSize(g);
    res.LocalVertexSize = localVertexSize;

    bool *isUpdate;
    CUDACHECK(cudaMallocManaged(&isUpdate, sizeof(bool)));
    *isUpdate = false;
    cc_process_messages<<<g->gridSize, g->blockSize>>>(g->active, g->updateByMessage, localVertexSize, isUpdate);
    CUDACHECK(cudaDeviceSynchronize());

    int step = 0;
    while (*isUpdate) {
        *isUpdate = false;
        cc_exec<<<g->gridSize, g->blockSize>>>(g->active, values->ccValue, g->index, g->dst, localVertexSize,
                                               step, isUpdate);
        CUDACHECK(cudaDeviceSynchronize());
//        printf("Inc --- step:%d\n", step);
        step++;
    }

    end = clock();
    res.CalTime = (float) (end - start) / CLOCKS_PER_SEC;

    start = clock();
    cc_Mirror2MasterSend(values, g, &comm->comm, &comm->s, workerId, workerNum, localVertexSize, &res);
    cc_Master2MirrorSend(values, g, &comm->comm, &comm->s, workerId, workerNum, localVertexSize, &res);
    end = clock();
    res.SendTime = (float) (end - start) / CLOCKS_PER_SEC;

    cudaFree(isUpdate);

    res.VisitedSize = countActive(g->active, localVertexSize, g->gridSize, g->blockSize);
    return res;
}

