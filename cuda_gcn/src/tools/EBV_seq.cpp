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
int *connect;

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

void init(char* graph, int GID=-1) {
    maxn = -1;
    if (strcmp(graph, "products") == 0) {maxn = 2449029; maxm = 64308169; workerNum = 3;}
//    if (strcmp(graph, "products") == 0) {maxn = 2449029; maxm = 32146758; workerNum = 4;}
//    if (strcmp(graph, "products") == 0) {maxn = 2449029; maxm = 32161411; workerNum = 4;}
    if (strcmp(graph, "friend") == 0) {
		if (GID == -1) {maxn = 65608366; maxm = 1871675501; workerNum = 4;}
		if (GID == 0) {maxn = 65608366; maxm = 467919959; workerNum = 8;}
		if (GID == 1) {maxn = 65608366; maxm = 467881646; workerNum = 8;}
		if (GID == 2) {maxn = 65608366; maxm = 467968343; workerNum = 8;}
		if (GID == 3) {maxn = 65608366; maxm = 467905553; workerNum = 8;}
	}

    if (strcmp(graph, "papers100M-bin") == 0) {
		if (GID == -1) {maxn = 111059956; maxm = 1726745828LL; workerNum = 4;}
		if (GID == 0) {maxn = 111059956; maxm = 431672343; workerNum = 8;}
		if (GID == 1) {maxn = 111059956; maxm = 431693535; workerNum = 8;}
		if (GID == 2) {maxn = 111059956; maxm = 431671924; workerNum = 8;}
		if (GID == 3) {maxn = 111059956; maxm = 431708026; workerNum = 8;}
	}
//    if (strcmp(graph, "papers100M-bin") == 0) {maxn = 111059956; maxm = 431672343LL; workerNum = 8;}
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
    connect = new int [workerNum];
    printf("init ok!\n");
}

int main(int argc, char** argv) {
    bool sort_by_degree = true;
    bool written = true;
    bool sc = true;

//    char *graph = "products";
//    char *graph = "friend";
	int GID = -1;
	if (argc > 1) sscanf(argv[1], "%d", &GID);
	printf("GID: %d\n", GID);
    char *graph = "friend";

    double alpha = 0.01;
    double beta = 0.01;
    init(graph, GID);

    clock_t start, finish;
    double duration;
    start = clock();

    FILE *fp;
	char path[100];
    if (sc) {
        if (strcmp(graph, "products") == 0) fp = fopen("/BIGDATA1/acict_zguan_1/zpltys/products/edgelist.txt", "r");
		if (strcmp(graph, "friend") == 0) {
			if (GID == -1) fp = fopen("/BIGDATA1/acict_zguan_1/zpltys/friend/edgelist.txt", "r");
			else {
				sprintf(path, "/BIGDATA1/acict_zguan_1/zpltys/friend/EBV_%d/G.%d", 4, GID);
				printf("read file from: %s\n", path);
				fp = fopen(path, "r");
			}
		}
        if (strcmp(graph, "papers100M-bin") == 0) {
			if (GID == -1) fp = fopen("/BIGDATA1/acict_zguan_1/zpltys/paper/papers100M-bin/raw/edgelist.txt", "r");
			else {
				sprintf(path, "/BIGDATA1/acict_zguan_1/zpltys/paper/%s/raw/EBV_%d/G.%d", graph, 4, GID);
				printf("read file from: %s\n", path);
				fp = fopen(path, "r");
			}
		}
    } else {
        if (strcmp(graph, "products") == 0) fp = fopen("/slurm/zhangshuai/GNN_Dataset/products/raw/edgelist.txt", "r");
//    if (strcmp(graph, "products") == 0) fp = fopen("/slurm/zhangshuai/GNN_Dataset/products/raw/EBV_2/G.1", "r");
//    if (strcmp(graph, "friend") == 0) fp = fopen("/slurm/zhangshuai/graphs/graph/friendster.txt", "r");
        if (strcmp(graph, "friend") == 0) fp = fopen("/slurm/zhangshuai/GNN_Dataset/friend/raw/EBV_4/G.1", "r");

        if (strcmp(graph, "papers100M-bin") == 0) fp = fopen("/slurm/zhangshuai/GNN_Dataset/papers100M-bin/raw/edgelist.txt", "r");
//    if (strcmp(graph, "papers100M-bin") == 0) fp = fopen("/slurm/zhangshuai/GNN_Dataset/papers100M-bin/raw/EBV_4/G.0", "r");
    }

    long long i, j;
    int u, v, temp;
    for (i = 0; i < workerNum; i++) {
        cou[i] = 0;
        vertex_count[i] = 0;
        for (j = 0; j < maxn; j++) has[i][j] = false;
    }
    maxCou = maxm / workerNum;

    FILE *G[workerNum], *Master[workerNum], *Mirror[workerNum];

    if (written) {
        if (sc) {
			if (GID == -1) {
				for (i = 0; i < workerNum; i++) {
					sprintf(path, "EBV_%d/G.%d", workerNum, i);
					G[i] = fopen(path, "w+");
					sprintf(path, "EBV_%d/Master.%d", workerNum, i);
					Master[i] = fopen(path, "w+");
					sprintf(path, "EBV_%d/Mirror.%d", workerNum, i);
					Mirror[i] = fopen(path, "w+");
				}
			} else {
				for (i = 0; i < workerNum; i++) {
					sprintf(path, "EBV_%d/%d/G.%d", 4, GID, i);
					G[i] = fopen(path, "w+");
					sprintf(path, "EBV_%d/%d/Master.%d", 4, GID, i);
					Master[i] = fopen(path, "w+");
					sprintf(path, "EBV_%d/%d/Mirror.%d", 4, GID, i);
					Mirror[i] = fopen(path, "w+");
				}
			}
        } else {
            for (i = 0; i < workerNum; i++) {
                sprintf(path, "/slurm/zhangshuai/GNN_Dataset/%s/raw/EBV_%d/G.%d", graph, workerNum, i);
                G[i] = fopen(path, "w+");
                sprintf(path, "/slurm/zhangshuai/GNN_Dataset/%s/raw/EBV_%d/Master.%d", graph, workerNum, i);
                Master[i] = fopen(path, "w+");
                sprintf(path, "/slurm/zhangshuai/GNN_Dataset/%s/raw/EBV_%d/Mirror.%d", graph, workerNum, i);
                Mirror[i] = fopen(path, "w+");
            }
        }
    }

// ------------------------------------------------------
    for (i = 0; i < maxn; i++) degree[i] = 0;
//    printf("???\n");

    int edge_count = 0;
    for (i = 0; i < maxm; i++) {
//        printf("i:%lld\n", i);
        fscanf(fp, "%d%d", &u, &v);
        e[edge_count].u = u;
        e[edge_count].v = v;
        edge_count++;
        degree[u]++;
        degree[v]++;

        if (i % 1000000 == 0) printf("read %lld/%lld lines\n", i, maxm);
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
        u = e[i].u;
        v = e[i].v;
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
        max_vertex = max(max_vertex, (double) vertex_count[i]);
        sum_vertex += vertex_count[i];
    }
    replication /= maxn;
    printf("%s part! ------- vertex imbalance: %.6f, edge imbalance: %.6f, replication: %.6f\n", graph,
           max_vertex / sum_vertex * workerNum, imbalance, replication);
    cout << "maxCou:" << maxCou << "  maxm:" << maxm << "  workerNum:" << workerNum << endl;
    // cout << "imbalance: " << imbalance << "  replication: " << replication << endl;

    for (i = 0; i < workerNum; i++) connect[i] = 0;
    for (u = 0; u < maxn; u++) {
        int len = rep[u].size();
        if (len <= 1) continue;
        int master = rep[u][len - 1];
        if (written) {
            fprintf(Master[master], "%d", u);
            for (i = 0; i < len - 1; i++) {
                fprintf(Mirror[rep[u][i]], "%d %d\n", u, master);
                fprintf(Master[master], " %d", rep[u][i]);
            }
            fprintf(Master[master], "\n");
        }
        connect[master] += len - 1;
        for (i = 0; i < len - 1; i++) connect[rep[u][i]]++;

        rep[u].clear();
    }
    long long all_connect = 0;
    for (i = 0; i < workerNum; i++) {
        all_connect += connect[i];
        printf("Worker %d, connect:%d\n", i, connect[i]);
    }
    printf("All connect:%lld\n", all_connect / 2);

    if (written) {
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
