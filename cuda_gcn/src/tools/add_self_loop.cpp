//
// Created by zplty on 2023/6/29.
//

#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

int maxn, workerNum;
ll maxm;

// ---------------------------
int *e, *nex, *fa, ee = 0;

bool addedge(int u, int v) {
    int i = fa[v];
    bool directed = false;
    while (i != -1) {
        if (e[i] == u) {
            directed = true;
            break;
        }
        i = nex[i];
    }

    nex[ee] = fa[u]; fa[u] = ee; e[ee] = v; ee++;
    return directed;
}

void init(char* graph) {
    maxn = -1;
    if (strcmp(graph, "products") == 0) {maxn = 2449029; maxm = 64308169;}
//    if (strcmp(graph, "products") == 0) {maxn = 2449029; maxm = 32146758; workerNum = 4;}
//    if (strcmp(graph, "products") == 0) {maxn = 2449029; maxm = 32161411; workerNum = 4;}
    if (strcmp(graph, "friend") == 0) {maxn = 65608366; maxm = 1806067135;}

    if (strcmp(graph, "papers100M-bin") == 0) {maxn = 111059956; maxm = 1726745828LL;}
//    if (strcmp(graph, "papers100M-bin") == 0) {maxn = 111059956; maxm = 431672343LL;}
    if (maxn == -1) {
        printf("error graph!\n");
        exit(1);
    }
    fa = new int [maxn];
    e = new int [maxm];
    nex = new int [maxm];
    for (int i = 0; i < maxn; i++) fa[i] = -1;
}

int main(int argc, char** argv) {

//    char *graph = "products";
    char *graph = "friend";
//    char *graph = "papers100M-bin";

    init(graph);

    FILE *fp;
    if (strcmp(graph, "products") == 0) fp = fopen("/slurm/zhangshuai/GNN_Dataset/products/raw/edgelist.txt", "r");
//    if (strcmp(graph, "products") == 0) fp = fopen("/slurm/zhangshuai/GNN_Dataset/products/raw/EBV_2/G.1", "r");
//    if (strcmp(graph, "friend") == 0) fp = fopen("/slurm/zhangshuai/graphs/graph/friendster.txt", "r");
    if (strcmp(graph, "friend") == 0) fp = fopen("/slurm/zhangshuai/GNN_Dataset/friend/raw/EBV_4/G.1", "r");

    if (strcmp(graph, "papers100M-bin") == 0) fp = fopen("/slurm/zhangshuai/GNN_Dataset/papers100M-bin/raw/edgelist.txt", "r");
//    if (strcmp(graph, "papers100M-bin") == 0) fp = fopen("/slurm/zhangshuai/GNN_Dataset/papers100M-bin/raw/EBV_4/G.0", "r");

    long long i, j;
    int u, v;

    char path[100];

    int edge_count = 0;
    for (i = 0; i < maxm; i++) {
//        printf("i:%lld\n", i);
        fscanf(fp, "%d%d", &u, &v);
        bool directed = addedge(u, v);
        if (directed) {
            printf("%d -> %d, directed!\n", u, v);
            return 0;
        }

        if (i % 1000000 == 0) printf("read %lld/%lld lines\n", i, maxm);
    }
}
