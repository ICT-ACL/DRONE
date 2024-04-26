//
// Created by zplty on 2023/4/7.
//

#ifndef GCN_GCN_H
#define GCN_GCN_H

typedef void* GCNHandle;

GCNHandle getHandle(int workerId, int workerNum, int globalVertexSize);

void run(GCNHandle handle);

#endif //GCN_GCN_H
