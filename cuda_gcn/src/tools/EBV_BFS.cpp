#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

int maxn, workerNum;
const int workerPerNode = 8;
ll maxm;

// double weight[workerNum];
// bool has[workerNum][maxn];
// int cou[workerNum];
double *weight;
bool **has;
int **has_node;
int *cou, *degree;
int maxCou;
int *vertex_count;

int *inner_connect, *outer_connect;

int *depth;
vector<int> *nex;


// ---------------------------
struct edge {
    int u, v;
    float weight;
}*e;
// int degree[maxn];
bool cmp (edge a, edge b) {
    return a.weight < b.weight;
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

void multiSource_BFS(int mod = 10000)
{
    mod = maxn / workerNum / 5;
    queue<int> q;
    for (int i = 0; i < maxn; i++) {
        if (i % mod == 0) {
            depth[i] = 0;
            q.push(i);
        }
        else depth[i] = 10000000;
    }

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int i = 0; i < nex[u].size(); i++) {
            int v = nex[u][i];

            if (depth[v] > depth[u] + 1) {
                depth[v] = depth[u] + 1;
                q.push(v);
            }
        }
    }
}

void init(char* graph) {
    maxn = -1;
    if (strcmp(graph, "products") == 0) {maxn = 2449029; maxm = 64308169; workerNum = 8;}
    if (strcmp(graph, "friend") == 0) {maxn = 65608366; maxm = 1806067135; workerNum = 32;}

    if (maxn == -1) {
        printf("error graph!\n");
        exit(1);
    }
    e = new edge[maxm];
    weight = new double[workerNum];
    has = new bool*[workerNum];
    for (int i = 0; i < workerNum; i++) has[i] = new bool[maxn];
    has_node = new int*[workerNum / workerPerNode];
    for (int i = 0; i < workerNum / workerPerNode; i++) has_node[i] = new int[maxn];

    cou = new int[workerNum];
    degree = new int[maxn];
    rep = new vector<int> [maxn];
    vertex_count = new int[workerNum];

    depth = new int [maxn];
    nex = new vector<int> [maxn];

    inner_connect = new int [workerNum];
    outer_connect = new int [workerNum];
    for (int i = 0; i < workerNum; i++) {
        inner_connect[i] = 0;
        outer_connect[i] = 0;
    }
}

int main(int argc, char** argv) {
    bool sort_by_degree = true;
    bool written = false;

//    char *graph = "products";
    char *graph = "friend";

    double alpha = 1.0;
    double beta = 1.0;
//    double gama = 1.0;
    double gama = 0.0;
    init(graph);

    clock_t start, finish;
    double duration;
    start = clock();

    FILE *fp;
    if (strcmp(graph, "products") == 0) fp = fopen("/slurm/zhangshuai/GNN_Dataset/products/raw/edgelist.txt", "r");
//    if (strcmp(graph, "products") == 0) fp = fopen("/BIGDATA1/acict_zguan_1/zpltys/products/edgelist.txt", "r");
    if (strcmp(graph, "friend") == 0) fp = fopen("/slurm/zhangshuai/graphs/graph/friendster.txt", "r");

    int i, u, v, j, temp;
    for (i = 0; i < workerNum; i++) {
        cou[i] = 0; vertex_count[i] = 0;
        for (j = 0; j < maxn; j++) has[i][j] = false;
    }
    for (i = 0; i < workerNum / workerPerNode; i++)
        for (j = 0; j < maxn; j++)
            has_node[i][j] = false;
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
        nex[u].push_back(v);
        nex[v].push_back(u);
        edge_count++;
        degree[u]++; degree[v]++;
    }
    maxm = edge_count;

//    multiSource_BFS();

    for (i = 0; i < maxm; i++) {
        if (sort_by_degree) e[i].weight = degree[e[i].u] + degree[e[i].v];
//        if (sort_by_degree) e[i].weight = min(depth[e[i].u], depth[e[i].v]) + 0.1 * max(depth[e[i].u], depth[e[i].v]);
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
            if (!has[j][u]) weight[j] += gama * has_node[j / workerPerNode][u] / workerPerNode;
            if (!has[j][v]) weight[j] += gama * has_node[j / workerPerNode][v] / workerPerNode;
            if (!has[j][u]) weight[j] -= 1;
            if (!has[j][v]) weight[j] -= 1;
//          weight[j] -= 1;

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
            has_node[assign / workerPerNode][u]++;
        }
        if (!has[assign][v]) {
            has[assign][v] = true;
            rep[v].push_back(assign);
            vertex_count[assign]++;
            has_node[assign / workerPerNode][v]++;
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

    int nodes = workerNum / workerPerNode;
    for (u = 0; u < maxn; u++) {
        int max_node_count = 0;
        for (i = 0; i < nodes; i++) {
            if (max_node_count < has_node[i][u]) {
                max_node_count = has_node[i][u];
            }
        }
        if (max_node_count == 0) continue;

        int len = rep[u].size();
        int masterID = -1;
        for (i = len - 1; i >= 0; i--) {
            if (has_node[rep[u][i] / workerPerNode][u] == max_node_count) {
                masterID = rep[u][i];
                break;
            }
        }
        for (i = 0; i < len; i++) {
            int mirrorID = rep[u][i];
            if (mirrorID == masterID) continue;
            if (mirrorID / workerPerNode == masterID / workerPerNode) {
                inner_connect[mirrorID]++;
                inner_connect[masterID]++;
            } else {
                outer_connect[mirrorID]++;
                outer_connect[masterID]++;
            }
        }
    }

    long long sum_inner = 0, sum_outer = 0;

    for (i = 0; i < workerNum; i++) {
        printf("Worker %d ---- inner_connect:%d, outer_connect:%d\n", i, inner_connect[i], outer_connect[i]);
        sum_inner += inner_connect[i];
        sum_outer += outer_connect[i];
    }
    printf("Sum ---- sum inner:%lld, sum outer:%lld\n", sum_inner / 2, sum_outer / 2);

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
