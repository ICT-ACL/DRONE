#include "stdio.h"
#include "cuda_runtime.h"
#include "nccl.h"
#include <iostream>
#include "time.h"
extern "C" {
#include "sssp.h"
}


__device__ float fatomicMin(float *addr, float value) {
    float ret = *addr;
    float old = ret, assumed;
    if (old <= value) return old;
    do {
        assumed = old;
        old = atomicCAS((unsigned int *) addr, __float_as_int(assumed), __float_as_int(value));

    } while (old != assumed);
    return ret;
}

__global__ void sssp_init(int *active, float *distance, int startLocalID, int localVertexSize, bool *isUpdate) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        if (id == startLocalID) {
            active[id] = 0;
            distance[id] = 0.0;
            *isUpdate = true;
        } else {
            active[id] = -1;
            distance[id] = 1e8;
        }
        id += stride;
    }
}

__global__ void sssp_exec(int *active, float *distance, int *index, int *dst, float *edgeLen, int localVertexSize, int step,
          bool *isUpdate) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        if (active[id] != step) {
            id += stride;
            continue;
        }
//        printf("new active vertex:%d, distance:%f, active:%d\n", id, distance[id], active[id]);

        float nowDis = distance[id];
        for (int i = index[id]; i < index[id + 1]; i++) {
            int v = dst[i];

//            if (edgeLen[i] != 1) {
//                printf("i:%d, edgeLen:%f \n", i, edgeLen[i]);
//            }
            float newDis = nowDis + edgeLen[i];

//            if (step == 1) printf("new search vertex:%d, oldDis:%f\n", v, distance[v]);

            float old = fatomicMin(&distance[v], newDis);
            if (old > newDis + 1e-6) {
                active[v] = step + 1;
//                if (step == 1) printf("active: %d, old:%f, newDis:%f\n", v, old, newDis);
                *isUpdate = true;
            }
//            if (step == 1) printf("new search vertex:%d, newDis:%f, old:%f, nowDis:%f, edgeLen:%f\n", v, newDis, old, nowDis, edgeLen[i]);
        }
        id += stride;
    }
}

__global__ void sssp_cal_mirror_sendLen(int *sendLen, int *Mirror2Worker, int *active, int localVertexSize) {
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


__global__ void sssp_cal_master_sendLen(int *sendLen, int *MasterWorkerIndex, int *Master2Workers, int *active, int *updateByMessage,
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

__global__ void sssp_generate_sendBuff_mirror(float *sendBuffFloat, int *sendBuffInt, int *sendIndex, int *sendDiff,
                                         int *Mirror2Worker, int *Local2Global, float *distance, int *active,
                                         int localVertexSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        if (active[id] <= 0 || Mirror2Worker[id] == -1) {
            id += stride;
            continue;
        }
        int globalId = Local2Global[id];
        float newDis = distance[id];
        int worker = Mirror2Worker[id];
        int *address = sendDiff + worker;
        int diff = atomicAdd(address, 1);

        sendBuffInt[sendIndex[worker] + diff] = globalId;
        sendBuffFloat[sendIndex[worker] + diff] = newDis;

        id += stride;
    }
}

__global__ void sssp_generate_sendBuff_master(float *sendBuffFloat, int *sendBuffInt, int *sendIndex, int *sendDiff,
                                         int *MasterWorkerIndex, int *Master2Workers, int *Local2Global,
                                         float *distance,
                                         int *active, int *updateByMessage, int localVertexSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < localVertexSize) {
        if (active[id] <= 0 && updateByMessage[id] == 0) {
            id += stride;
            continue;
        }
        int globalId = Local2Global[id];
        float newDis = distance[id];

        for (int i = MasterWorkerIndex[id]; i < MasterWorkerIndex[id + 1]; i++) {
            int worker = Master2Workers[i];
            int *address = sendDiff + worker;
            int diff = atomicAdd(address, 1);
            sendBuffInt[sendIndex[worker] + diff] = globalId;
            sendBuffFloat[sendIndex[worker] + diff] = newDis;
        }

        id += stride;
    }
}

__global__ void sssp_process_recvBuff(float *recvBuffFloat, int *recvBuffInt, int recvLen, int *Global2Local, float *distance,
                 int *updateByMessage) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    while (id < recvLen) {
        int localId = Global2Local[recvBuffInt[id]];
        float newDis = recvBuffFloat[id];

        if (localId == -1) {
            printf("error global id :%d!\n", recvBuffInt[id]);
            id += stride;
            continue;
        }

        float old = fatomicMin(&distance[localId], newDis);
        if (old > newDis) {
            updateByMessage[localId] = 1;
        }

        id += stride;
    }
}

//__global__ void CUDA_FILL_INT(int *array, int val, int maxLen) {
//    int id = threadIdx.x + blockIdx.x * blockDim.x;
//    int stride = gridDim.x * blockDim.x;
//    while (id < maxLen) {
//        array[id] = val;
//        id += stride;
//    }
//}

void sssp_Mirror2MasterSend(SSSPValues *values, Graph *g, ncclComm_t *comm, cudaStream_t *s, int workerId, int workerNum,
                       int localVertexSize, Response *res) {
//    printf("Call sssp_Mirror2MasterSend\n");
//    int device;
//    cudaGetDevice(&device);
//    printf("sssp_Mirror2MasterSend: device: %d\n", device);

    int *sendLen, *recvLen;
    CUDACHECK(cudaMallocManaged(&sendLen, sizeof(int) * workerNum));
    cudaMallocManaged(&recvLen, sizeof(int) * workerNum);
    for (int i = 0; i < workerNum; i++) {
        sendLen[i] = 0;
    }

//    printf("Before invoke sssp_cal_mirror_sendLen\n");

    sssp_cal_mirror_sendLen<<<g->gridSize, g->blockSize>>>(sendLen, g->Mirror2Worker, g->active, localVertexSize);

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

    sssp_generate_sendBuff_mirror<<<g->gridSize, g->blockSize>>>(values->sendDis, values->sendID, sendIndex, sendDiff,
                                                            g->Mirror2Worker, g->Local2Global, values->distance,
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
        ncclSend(values->sendDis + sendIndex[i], sendLen[i], ncclFloat, i, *comm, *s);
        ncclRecv(values->recvDis + recvIndex[i], recvLen[i], ncclFloat, i, *comm, *s);
    }
    ncclGroupEnd();
    cudaStreamSynchronize(*s);

    int sendSize = sendIndex[workerNum];
    int recvSize = recvIndex[workerNum];

    res->Mirror2MasterSendSize = sendSize;
    res->Mirror2MasterRecvSize = recvSize;

    sssp_process_recvBuff<<<g->gridSize, g->blockSize>>>(values->recvDis, values->recvID, recvSize, g->Global2Local,
                                                    values->distance, g->updateByMessage);
    CUDACHECK(cudaDeviceSynchronize());

    cudaFree(sendLen);
    cudaFree(recvLen);
    cudaFree(sendIndex);
    cudaFree(sendDiff);
    cudaFreeHost(recvIndex);
}

void sssp_Master2MirrorSend(SSSPValues *values, Graph *g, ncclComm_t *comm, cudaStream_t *s, int workerId, int workerNum,
                       int localVertexSize, Response *res) {
    int *sendLen, *recvLen;
    cudaMallocManaged(&sendLen, sizeof(int) * workerNum);
    cudaMallocManaged(&recvLen, sizeof(int) * workerNum);
    for (int i = 0; i < workerNum; i++) {
        sendLen[i] = 0;
    }
    sssp_cal_master_sendLen<<<g->gridSize, g->blockSize>>>(sendLen, g->MasterWorkerIndex, g->Master2Workers, g->active,
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

    sssp_generate_sendBuff_master<<<g->gridSize, g->blockSize>>>(values->sendDis, values->sendID, sendIndex, sendDiff,
                                                            g->MasterWorkerIndex, g->Master2Workers,
                                                            g->Local2Global,
                                                            values->distance,
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
        ncclSend(values->sendDis + sendIndex[i], sendLen[i], ncclFloat, i, *comm, *s);
        ncclRecv(values->recvDis + recvIndex[i], recvLen[i], ncclFloat, i, *comm, *s);
    }
    ncclGroupEnd();
    CUDACHECK(cudaStreamSynchronize(*s));

    int sendSize = sendIndex[workerNum];
    int recvSize = recvIndex[workerNum];

    res->Master2MirrorSendSize = sendSize;
    res->Master2MirrorRecvSize = recvSize;

    sssp_process_recvBuff<<<g->gridSize, g->blockSize>>>(values->recvDis, values->recvID, recvSize, g->Global2Local,
                                                    values->distance, g->updateByMessage);
    CUDACHECK(cudaDeviceSynchronize());

    cudaFree(sendLen);
    cudaFree(recvLen);
    cudaFree(sendIndex);
    cudaFree(sendDiff);
    cudaFreeHost(recvIndex);
}

//__global__ void count(int *active, int localVertexSize, int *res) {
//    int id = threadIdx.x + blockIdx.x * blockDim.x;
//    int stride = gridDim.x * blockDim.x;
//    while (id < localVertexSize) {
//        if (active[id] <= 0) {
//            atomicAdd(res, 1);
//        }
//        id += stride;
//    }
//}
//
//int countActive(int *active, int localVertexSize, int gridSize, int blockSize) {
//    int *res_h, *res_d;
//    cudaMalloc(&res_d, sizeof(int));
//    CUDACHECK(cudaMallocHost(&res_h, sizeof(int)));
//    *res_h = 0;
//    cudaMemcpy(res_d, res_h, sizeof(int), cudaMemcpyHostToDevice);
//    count<<<gridSize, blockSize>>>(active, localVertexSize, res_d);
//    CUDACHECK(cudaDeviceSynchronize());
//    cudaMemcpy(res_h, res_d, sizeof(int), cudaMemcpyDeviceToHost);
//
//    int res = *res_h;
//    cudaFree(res_d);
//    cudaFreeHost(res_h);
//    return res;
//}

Response SSSP_PEVal(Graph *g, SSSPValues *values, int startId, int workerId, int workerNum, Comm *comm) {
    clock_t start, end;
    start = clock();

    cudaSetDevice(g->GID);
    printf("C call SSSP_PEVal!\n");
    Response res;
    int startLocalID;
    cudaMemcpy(&startLocalID, &g->Global2Local[startId], sizeof(int), cudaMemcpyDeviceToHost);

    int localVertexSize = getLocalVertexSize(g);
    cudaMalloc(&g->active, sizeof(int) * localVertexSize);
    cudaMalloc(&values->distance, sizeof(float) * localVertexSize);
    cudaMalloc(&g->updateByMessage, sizeof(int) * localVertexSize);

    bool *isUpdate;
    cudaMallocManaged(&isUpdate, sizeof(bool));
    *isUpdate = false;

    sssp_init<<<g->gridSize, g->blockSize>>>(g->active, values->distance, startLocalID, localVertexSize, isUpdate);
    cudaDeviceSynchronize();

    res.LocalVertexSize = localVertexSize;

    int step = 0;
    while (*isUpdate) {
        *isUpdate = false;
        sssp_exec<<<g->gridSize, g->blockSize>>>(g->active, values->distance, g->index, g->dst, g->edgeLen,
                                                 localVertexSize,
                                                 step, isUpdate);
        cudaDeviceSynchronize();
        step++;
    }

//    ncclUniqueId id;
//    if (workerId == 0) {
//        ncclGetUniqueId(&id);
//        FILE *fp = fopen("./nccl.id", "w");
//        fwrite(&id, sizeof(id), 1, fp);
//        fclose(fp);
//    } else {
//        FILE *fp = fopen("./nccl.id", "r");
//        fread(&id, sizeof(id), 1, fp);
//        fclose(fp);
//    }

//    FILE *fp = fopen("./nccl.id", "r");
//    fread(&id, sizeof(id), 1, fp);
//    fclose(fp);
//    NCCLCHECK(ncclCommInitRank(&comm->comm, workerNum, id, workerId));

    cudaMalloc(&values->sendID, sizeof(int) * max(g->MirrorSize, g->MirrorWorkerSize));
    cudaMalloc(&values->sendDis, sizeof(float) * max(g->MirrorSize, g->MirrorWorkerSize));
    cudaMalloc(&values->recvID, sizeof(int) * max(g->MirrorSize, g->MirrorWorkerSize));
    cudaMalloc(&values->recvDis, sizeof(float) * max(g->MirrorSize, g->MirrorWorkerSize));

    printf("Buffer apply size:%d\n", max(g->MirrorSize, g->MirrorWorkerSize));
    printf("MirrorSize:%d, MasterSize:%d, MirrorWorkerSize:%d\n", g->MirrorSize, g->MasterSize,
           g->MirrorWorkerSize);

    end = clock();
    res.CalTime = (float) (end - start) / CLOCKS_PER_SEC;

    start = clock();
    printf("sssp_Mirror2MasterSend start!\n");
    sssp_Mirror2MasterSend(values, g, &comm->comm, &comm->s, workerId, workerNum, localVertexSize, &res);
    printf("sssp_Mirror2MasterSend ok!\n");
    sssp_Master2MirrorSend(values, g, &comm->comm, &comm->s, workerId, workerNum, localVertexSize, &res);
    printf("sssp_Master2MirrorSend ok!\n");
    end = clock();
    res.SendTime = (float) (end - start) / CLOCKS_PER_SEC;


    cudaFree(isUpdate);
    printf("countActive start!\n");
    res.VisitedSize = countActive(g->active, localVertexSize, g->gridSize, g->blockSize);
    printf("countActive ok!\n");
    return res;
}

__global__ void sssp_process_messages(int *active, int *updateByMessage, int maxLen, bool *isUpdate) {
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

Response SSSP_IncEVal(Graph *g, SSSPValues *values, int workerId, int workerNum, Comm *comm) {
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
    sssp_process_messages<<<g->gridSize, g->blockSize>>>(g->active, g->updateByMessage, localVertexSize, isUpdate);
    CUDACHECK(cudaDeviceSynchronize());
//    printf("Inc: sssp_process_messages ok!\n");

    int step = 0;
    while (*isUpdate) {
        *isUpdate = false;
//        printf("before sssp_exec\n");
        sssp_exec<<<g->gridSize, g->blockSize>>>(g->active, values->distance, g->index, g->dst, g->edgeLen,
                                                 localVertexSize,
                                                 step, isUpdate);
        CUDACHECK(cudaDeviceSynchronize());
//        printf("Inc --- step:%d\n", step);
        step++;
    }

    end = clock();
    res.CalTime = (float) (end - start) / CLOCKS_PER_SEC;

    start = clock();
    sssp_Mirror2MasterSend(values, g, &comm->comm, &comm->s, workerId, workerNum, localVertexSize, &res);
    sssp_Master2MirrorSend(values, g, &comm->comm, &comm->s, workerId, workerNum, localVertexSize, &res);
    end = clock();
    res.SendTime = (float) (end - start) / CLOCKS_PER_SEC;

    cudaFree(isUpdate);

    res.VisitedSize = countActive(g->active, localVertexSize, g->gridSize, g->blockSize);
    return res;
}

