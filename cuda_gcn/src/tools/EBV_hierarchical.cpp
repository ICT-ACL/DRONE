//
// Created by zplty on 2023/6/29.
//

#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

int maxn, workerNum;
ll maxm;

// double weight[workerNum];
// bool has[workerNum][maxn];
// int cou[workerNum];
double *weight;
bool **has;
int *cou, *degree;
int maxCou;
int *vertex_count;

// ---------------------------
struct edge {
    int u, v, degree;
}*e;
// int degree[maxn];
bool cmp (edge a, edge b) {
    return a.degree < b.degree;
}
// ---------------------------

int find_max() {
    double ma = weight[0];
    int ans = 0;
    for (int i = 1; i < workerNum; i++) {
        if (ma < weight[i]) {
            ma = weight[i];
            ans = i;
        }
    }
    return ans;
}

vector<int> *rep;

void init(char* graph) {
    maxn = -1;
//    if (strcmp(graph, "products") == 0) {maxn = 2449029; maxm = 64308169; workerNum = 2;}
//    if (strcmp(graph, "products") == 0) {maxn = 2449029; maxm = 32146758; workerNum = 4;}
    if (strcmp(graph, "products") == 0) {maxn = 2449029; maxm = 32161411; workerNum = 4;}

    if (strcmp(graph, "friend") == 0) {maxn = 65608366; maxm = 1806067135; workerNum = 32;}

    if (maxn == -1) {
        printf("error graph!\n");
        exit(1);
    }
    e = new edge[maxm];
    weight = new double[workerNum];
    has = new bool*[workerNum];
    for (int i = 0; i < workerNum; i++) has[i] = new bool[maxn];
    cou = new int[workerNum];
    degree = new int[maxn];
    rep = new vector<int> [maxn];
    vertex_count = new int[workerNum];
}

int main(int argc, char** argv) {
    bool sort_by_degree = true;
    bool written = false;

//    char *graph = "products";
    char *graph = "friend";

    double alpha = 0.01;
    double beta = 0.01;
    init(graph);

    clock_t start, finish;
    double duration;
    start = clock();

    FILE *fp;
//    if (strcmp(graph, "products") == 0) fp = fopen("/slurm/zhangshuai/GNN_Dataset/products/raw/edgelist.txt", "r");
    if (strcmp(graph, "products") == 0) fp = fopen("/slurm/zhangshuai/GNN_Dataset/products/raw/EBV_2/G.1", "r");
    if (strcmp(graph, "friend") == 0) fp = fopen("/slurm/zhangshuai/graphs/graph/friendster.txt", "r");

    int i, u, v, j, temp;
    for (i = 0; i < workerNum; i++) {
        cou[i] = 0; vertex_count[i] = 0;
        for (j = 0; j < maxn; j++) has[i][j] = false;
    }
    maxCou = maxm / workerNum;

    char path[100];
    FILE *G[workerNum], *Master[workerNum], *Mirror[workerNum];

    if (written) {
        for (i = 0; i < workerNum; i++) {
            sprintf(path, "/slurm/zhangshuai/GNN_Dataset/%s/raw/EBV_%d/G.%d", graph, workerNum, i);
            G[i] = fopen(path, "w+");
            sprintf(path, "/slurm/zhangshuai/GNN_Dataset/%s/raw/EBV_%d/Master.%d", graph, workerNum, i);
            Master[i] = fopen(path, "w+");
            sprintf(path, "/slurm/zhangshuai/GNN_Dataset/%s/raw/EBV_%d/Mirror.%d", graph, workerNum, i);
            Mirror[i] = fopen(path, "w+");
        }
    }

// ------------------------------------------------------
    for (i = 0; i < maxn; i++) degree[i] = 0;

    int edge_count = 0;
    for (i = 0; i < maxm; i++) {
        fscanf(fp, "%d%d", &u, &v);
        e[edge_count].u = u; e[edge_count].v = v;
        edge_count++;
        degree[u]++; degree[v]++;
    }
    maxm = edge_count;

    for (i = 0; i < maxm; i++) {
        if (sort_by_degree) e[i].degree = degree[e[i].u] + degree[e[i].v];

        // else e[i].degree = rand() % 10000;
    }

    if (sort_by_degree) sort(e, e + maxm, cmp);
// ------------------------------------------------------
    // for (i = 0; i<=10000; i += 100) {
    //     printf("i:%d ---- u:%d v:%d degree:%d\n",i, e[i].u, e[i].v, e[i].degree);
    // }

    for (i = 0; i < maxm; i++) {
        // fscanf(fp, "%d%d", &u, &v);
        u = e[i].u; v = e[i].v;
        for (j = 0; j < workerNum; j++) {
            weight[j] = 0;
            if (!has[j][u]) weight[j] -= 1;
            if (!has[j][v]) weight[j] -= 1;
            weight[j] = weight[j] - alpha * cou[j] / maxCou - beta * vertex_count[j] / maxn * workerNum;
        }
        //weight[hash] += alpha;
        int assign = find_max();
        if (written) fprintf(G[assign], "%d %d\n", u, v);
        cou[assign]++;
        // maxCou = max(maxCou, cou[assign]);
        if (!has[assign][u]) {
            has[assign][u] = true;
            rep[u].push_back(assign);
            vertex_count[assign]++;
        }
        if (!has[assign][v]) {
            has[assign][v] = true;
            rep[v].push_back(assign);
            vertex_count[assign]++;
        }
    }
    fclose(fp);

    maxCou = 0;
    for (i = 0; i < workerNum; i++) maxCou = max(maxCou, cou[i]);

    double replication = 0, imbalance = 1.0 * maxCou / maxm * workerNum;
    double max_vertex = 0, sum_vertex = 0;
    for (i = 0; i < workerNum; i++) {
        for (j = 0; j < maxn; j++) replication += has[i][j] ? 1 : 0;
        max_vertex = max(max_vertex, (double)vertex_count[i]);
        sum_vertex += vertex_count[i];
    }
    replication /= maxn;
    printf("%s part! ------- vertex imbalance: %.6f, edge imbalance: %.6f, replication: %.6f\n", graph, max_vertex / sum_vertex * workerNum, imbalance, replication);
    cout << "maxCou:" << maxCou << "  maxm:" << maxm << "  workerNum:" << workerNum << endl;
    // cout << "imbalance: " << imbalance << "  replication: " << replication << endl;

    if (written) {
        for (u = 0; u < maxn; u++) {
            int len = rep[u].size();
            if (len <= 1) continue;
            int master = rep[u][len - 1];
            fprintf(Master[master], "%d", u);
            for (i = 0; i < len - 1; i++) {
                fprintf(Mirror[rep[u][i]], "%d %d\n", u, master);
                fprintf(Master[master], " %d", rep[u][i]);
            }
            fprintf(Master[master], "\n");

            rep[u].clear();
        }
        for (i = 0; i < workerNum; i++) {
            fclose(G[i]);
            fclose(Master[i]);
            fclose(Mirror[i]);
        }
    }

    finish = clock();
    duration = (double) (finish - start) / CLOCKS_PER_SEC;
    printf("%f seconds\n", duration);
}
