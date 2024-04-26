#include <bits/stdc++.h>
using namespace std;

int maxn, workerNum;
int maxm;

// double weight[workerNum];
// bool has[workerNum][maxn];
// int cou[workerNum];
bool **has;
int **has_node;
int *cou, *degree;
int maxCou;
int *vertex_count;

int *inner_connect, *outer_connect;

int *depth;

const int first_part = 4;
const int second_part = 8;

// ---------------------------
struct edge {
	int u, v;
	int index;
	int assign;
}*e;
bool cmp (edge a, edge b) {
	return a.index < b.index;
}
// ---------------------------

vector<int> *rep;

void init(char* graph) {
	maxn = -1;
	if (strcmp(graph, "papers100M-bin") == 0) {maxn = 111059956; maxm = 1726745828;}
	if (strcmp(graph, "friend") == 0) {maxn = 65608366; maxm = 1871675501;}

	if (maxn == -1) {
		printf("error graph!\n");
		exit(1);
	}
	e = new edge[maxm];
	workerNum = first_part * second_part;
	has = new bool*[workerNum];
	for (int i = 0; i < workerNum; i++) has[i] = new bool[maxn];
	has_node = new int*[first_part];
	for (int i = 0; i < first_part; i++) has_node[i] = new int[maxn];

	cou = new int[workerNum];
	degree = new int[maxn];
	rep = new vector<int> [maxn];
	vertex_count = new int[workerNum];

	inner_connect = new int [workerNum];
	outer_connect = new int [workerNum];
	for (int i = 0; i < workerNum; i++) {
		inner_connect[i] = 0;
		outer_connect[i] = 0;
	}
}

int main(int argc, char** argv) {
	bool written = true;
	bool sc = true;

//    char *graph = "products";
//	char *graph = "papers100M-bin";
	char *graph = "friend";

	init(graph);

	clock_t start, finish;
	double duration;
	start = clock();

	int i, u, v, j, temp;
	for (i = 0; i < workerNum; i++) {
		cou[i] = 0; vertex_count[i] = 0;
		for (j = 0; j < maxn; j++) has[i][j] = false;
	}
	for (i = 0; i < first_part; i++)
		for (j = 0; j < maxn; j++)
			has_node[i][j] = false;
	maxCou = maxm / workerNum;

	char path[100];
	FILE *G[workerNum], *Master[workerNum], *Mirror[workerNum];

	if (written) {
		if (sc) {
			for (i = 0; i < workerNum; i++) {
				sprintf(path, "EBV_%d_merge/G.%d", workerNum, i);
				G[i] = fopen(path, "w+");
				printf("written to %s\n", path);
				sprintf(path, "EBV_%d_merge/Master.%d", workerNum, i);
				Master[i] = fopen(path, "w+");
				sprintf(path, "EBV_%d_merge/Mirror.%d", workerNum, i);
				Mirror[i] = fopen(path, "w+");
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

	FILE *fp;
	int ind_sum = 0;
	for (i = 0; i < first_part; i++) {
		for (j = 0; j < second_part; j++) {
			sprintf(path, "EBV_%d/%d/G.%d", first_part, i, j);
			fp = fopen(path, "r");

			int ind = 0;
			while (~fscanf(fp, "%d%d", &u, &v)) {
				e[ind_sum].u = u; e[ind_sum].v = v; e[ind_sum].assign = i * second_part + j;
				e[ind_sum++].index = ind++;
			}

			printf("read from %s finished!\n", path);
		}
	}
	printf("maxm: %d, ind_sum: %d\n", maxm, ind_sum);
	sort(e, e + maxm, cmp);

	for (i = 0; i < maxm; i++) {
		// fscanf(fp, "%d%d", &u, &v);
		u = e[i].u; v = e[i].v;

		int assign = e[i].assign;
		if (written) fprintf(G[assign], "%d %d\n", u, v);
		cou[assign]++;
		// maxCou = max(maxCou, cou[assign]);
		if (!has[assign][u]) {
			has[assign][u] = true;
			rep[u].push_back(assign);
			vertex_count[assign]++;
			has_node[assign / second_part][u]++;
		}
		if (!has[assign][v]) {
			has[assign][v] = true;
			rep[v].push_back(assign);
			vertex_count[assign]++;
			has_node[assign / second_part][v]++;
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

	int nodes = first_part;
	int *masterID;
	masterID = new int [maxn];
	for (u = 0; u < maxn; u++) {
		masterID[u] = -1;
		int max_node_count = 0;
		for (i = 0; i < nodes; i++) {
			if (max_node_count < has_node[i][u]) {
				max_node_count = has_node[i][u];
			}
		}
		if (max_node_count == 0) continue;

		int len = rep[u].size();
		for (i = len - 1; i >= 0; i--) {
			if (has_node[rep[u][i] / second_part][u] == max_node_count) {
				masterID[u] = rep[u][i];
				break;
			}
		}
		for (i = 0; i < len; i++) {
			int mirrorID = rep[u][i];
			if (mirrorID == masterID[u]) continue;
			if (mirrorID / second_part == masterID[u] / second_part) {
				inner_connect[mirrorID]++;
				inner_connect[masterID[u]]++;
			} else {
				outer_connect[mirrorID]++;
				outer_connect[masterID[u]]++;
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
			int master = masterID[u];
			fprintf(Master[master], "%d", u);
			for (i = 0; i < len; i++) {
				if (rep[u][i] == master) continue;
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