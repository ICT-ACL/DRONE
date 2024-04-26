//
// Created by zplty on 2023/3/29.
//

#include "bits/stdc++.h"
using namespace std;

int maxn, maxm;
int nparts;
int *in_degree, *out_degree;
bool **has;
int *part;

void init(char* graph) {
	if (strcmp(graph, "products") == 0) {maxn = 2449029; maxm = 64308169; nparts = 8;}
    if (strcmp(graph, "reddit") == 0) {maxn = 232965; maxm = 114615892; nparts = 4;}
	if (strcmp(graph, "friend") == 0) {maxn = 65608366; maxm = 1806067135; nparts = 16;}
	if (strcmp(graph, "papers100M-bin") == 0) {maxn = 111059956; maxm = 1615685872; nparts = 16;}

	printf("maxn:%d, maxm:%d\n", maxn, maxm);

	in_degree = new int [maxn];
	out_degree = new int [maxn];
	has = new bool* [nparts];
	for (int i = 0; i < nparts; i++) has[i] = new bool [maxn];
	part = new int [maxn];

	printf("init ok!\n");
}

//原始串行版本的图数据读入
void transfer2Ori()
{
    int maxn = 2449029;
    vector<vector<int> > edges;
    for (int i = 0; i < maxn; i++) edges.push_back(vector<int>());
    char fileName[] = "edge.csv";
    FILE *fp = fopen(fileName, "r");
    int u, v;
    char sep;
    while (~fscanf(fp, "%d%c%d", &u, &sep, &v)) {
        edges[u].push_back(v);
    }
    fclose(fp);

    fp = fopen("graph.txt", "w+");
    for (int i = 0; i < maxn; i++) {
        for (auto val : edges[i]) {
            fprintf(fp, "%d%c", val, ' ');
        }
        fprintf(fp, "%c", '\n');
    }
}

void generateInOutDegree()
{
    for (int i = 0; i < maxn; i++) {
        in_degree[i] = 0; out_degree[i] = 0;
    }

    FILE *fp = fopen("edgelist.txt", "r");
    int u, v;
    while (~fscanf(fp, "%d%d", &u, &v)) {
        in_degree[u]++; out_degree[v]++;
    }
    fclose(fp);

    fp = fopen("in_degree.txt", "w+");
    for (int i = 0; i < maxn; i++) fprintf(fp, "%d\n", in_degree[i]);
    fclose(fp);

    fp = fopen("out_degree.txt", "w+");
    for (int i = 0; i < maxn; i++) fprintf(fp, "%d\n", out_degree[i]);
    fclose(fp);
}

void transfer2DRONE(bool add_self_loop = true)
{
    char fileName[] = "edge.csv";
    FILE *fp = fopen(fileName, "r");
    FILE *out;
	if (add_self_loop) out = fopen("edgelist_loop.txt", "w+");
	else out = fopen("edgelist_noloop.txt", "w+");
    int u, v;
    char sep;
    printf("hello!\n");
    for (int i = 0; i < maxn; i++) {
        in_degree[i] = 0;
        out_degree[i] = 0;
    }
    while (~fscanf(fp, "%d%c%d", &u, &sep, &v)) {
        fprintf(out, "%d %d\n", u, v);
        out_degree[u]++;
        in_degree[v]++;
        if (u >= maxn || v >= maxn) {
            printf("u:%d v:%d\n", u, v);
        }
    }
	if (add_self_loop) {
		for (int i = 0; i < maxn; i++) {
			fprintf(out, "%d %d\n", i, i);
			in_degree[i]++; out_degree[i]++;
		}
	}
    fclose(out);
    fclose(fp);

    fp = fopen("in_degree.txt", "w+");
    for (int i = 0; i < maxn; i++) fprintf(fp, "%d\n", in_degree[i]);
    fclose(fp);

    fp = fopen("out_degree.txt", "w+");
    for (int i = 0; i < maxn; i++) fprintf(fp, "%d\n", out_degree[i]);
    fclose(fp);
}

void transfer2METIS_input()
{
	vector<vector<int> > arr;
    for (int i = 0; i < maxn; i++) arr.push_back(vector<int>());
    char fileName[] = "edgelist.txt";
//	char fileName[] = "friendster.txt";
    FILE *fp = fopen(fileName, "r");
    int u, v;
    char sep;
	int m = 0;
    while (~fscanf(fp, "%d%d", &u, &v)) {
		if (u == v) continue;
		bool has_edge = false;
		for (int i = 0; i < arr[u].size(); i++) if (arr[u][i] == v) {
			has_edge = true;
			break;
		}
		if (!has_edge) {
			m++;
			arr[u].push_back(v);
			arr[v].push_back(u);

			if (m % 100000 == 0) printf("read %d edges!\n", m);
		}
    }
    fclose(fp);

    FILE* fout = fopen("graph_metis_input.txt", "w+");
    fprintf(fout, "%d %d\n", maxn, m);

    for (int i = 0; i < maxn; i++) {
		vector<int>::iterator it;
		for (it = arr[i].begin(); it != arr[i].end(); it++) {
			fprintf(fout, "%d ", (*it) + 1);
		}
		if (i % 10000 == 0) printf("write %d lines\n", i);
		fprintf(fout, "\n");

//        for (auto val : edges[i]) {
//            fprintf(fout, "%d%c", val + 1, ' ');
//        }
//        fprintf(fout, "\n");
    }
    fclose(fout);
}

void METIS2DRONE()
{
	char path[100];
	sprintf(path, "graph_metis_input.txt.part.%d", nparts);
    FILE* fp = fopen(path, "r");
    int i, p, u, v;
    char sep;
    for (i = 0; i < maxn; i++) {
        fscanf(fp, "%d", &p);
        part[i] = p;
    }
    fclose(fp);

    for (i = 0; i < nparts; i++) {
        for (u = 0; u < maxn; u++) has[i][u] = false;
    }

    fp = fopen("edgelist.txt", "r");
    FILE *G[nparts], *Master[nparts], *Mirror[nparts];
    for (i = 0; i < nparts; i++) {
        char fileName[1000];
        sprintf(fileName, "product_%d_METIS/G.%d", nparts, i);
        G[i] = fopen(fileName, "w+");

        sprintf(fileName, "product_%d_METIS/Master.%d", nparts, i);
        Master[i] = fopen(fileName, "w+");

        sprintf(fileName, "product_%d_METIS/Mirror.%d", nparts, i);
        Mirror[i] = fopen(fileName, "w+");
    }

    while (~fscanf(fp, "%d %d", &u, &v)) {
        int workerID = part[v];
        has[workerID][u] = true;
        has[workerID][v] = true;
        fprintf(G[workerID], "%d %d\n", u, v);
    }
    fclose(fp);

    for (u = 0; u < maxn; u++) {
        int count = 0;
        for (i = 0; i < nparts; i++) {
            if (has[i][u]) count++;
        }

        if (count == 0) {
            printf("error! --- u:%d\n", u);
            exit(-1);
        }
        if (count > 1) {
            int masterID = part[u];
            assert(has[masterID][u]);
            fprintf(Master[masterID], "%d", u);
            for (p = 0; p < nparts; p++) {
                if (!has[p][u] || p == masterID) continue;
                fprintf(Master[masterID], " %d", p);
                fprintf(Mirror[p], "%d %d\n", u, masterID);
            }
            fprintf(Master[masterID], "\n");
        }
    }

    for (i = 0; i < nparts; i++) {
        fclose(G[i]);
        fclose(Master[i]);
        fclose(Mirror[i]);
    }
}

// for hep!
int* VertexConvert()
{
	char path[100];
	sprintf(path, "edgelist.txt.convert.txt");
	FILE *fp = fopen(path, "r");
	int* idmap = new int[maxn];
	int a, b;
	while (~fscanf(fp, "%d%d", &a, &b)) {
		idmap[a] = b;
	}
	return idmap;
}

//mpirun -np 2 ./DistributedNE /slurm/zhangshuai/GNN_Dataset/products/raw/edgelist.txt 2
typedef long long ll;
void convertDNE2PSHEP(char *partStrategy) {
	vector<int> *rep;
	rep = new vector<int> [maxn];
    FILE *fp;


    char path[100];
	int *convert;
	if (strcmp(partStrategy, "DNE") == 0) {
		sprintf(path, "edgelist.txt.%d.pedges", nparts);
	} else if (strcmp(partStrategy, "HEP") == 0) {
        sprintf(path, "edgelist.txt.edgepart.%d", nparts);
		convert = VertexConvert();
    } else {
		printf("error input!\n");
		exit(-1);
	}
    printf("read from %s\n", path);
    fp = fopen(path, "r");

    int i, u, v, j, assign;
    for (i = 0; i < nparts; i++) {
        for (j = 0; j < maxn; j++) has[i][j] = false;
    }

    FILE *G[nparts], *Master[nparts], *Mirror[nparts];

    for (i = 0; i < nparts; i++) {
        sprintf(path, "%s_%d/G.%d", partStrategy, nparts, i);
        G[i] = fopen(path, "w+");
        sprintf(path, "%s_%d/Master.%d", partStrategy, nparts, i);
        Master[i] = fopen(path, "w+");
        sprintf(path, "%s_%d/Mirror.%d", partStrategy, nparts, i);
        Mirror[i] = fopen(path, "w+");
    }

    int edge_count = 0;
    if (strcmp(partStrategy, "DNE") == 0) {
        char buf[1000];
        fgets(buf, 1000, fp);
        printf("skip: %s\n", buf);
    }
	// HEP会自动删除自环
//    for (i = 0; i < maxm; i++) {
//        fscanf(fp, "%d%d%d", &u, &v, &assign);
//		printf("u:%d, v:%d, assign:%d\n", u, v, assign);
	while (~fscanf(fp, "%d%d%d", &u, &v, &assign)) {
		if (strcmp(partStrategy, "HEP") == 0) {
			u = convert[u];
			v = convert[v];
		}

        if (!has[assign][u]) {
            has[assign][u] = true;
            rep[u].push_back(assign);
        }
        if (!has[assign][v]) {
            has[assign][v] = true;
            rep[v].push_back(assign);
        }
        fprintf(G[assign], "%d %d\n", u, v);
    }
    fclose(fp);

    for (u = 0; u < maxn; u++) {
        int len = rep[u].size();
		if (len == 1) {
			fprintf(G[rep[u][0]], "%d %d\n", u, u);
		}
        if (len <= 1) continue;
        int m_id = u % len;
        int master = rep[u][m_id];
		fprintf(G[master], "%d %d\n", u, u);
        fprintf(Master[master], "%d", u);
        for (i = 0; i < len; i++) {
            if (i != m_id) {
                fprintf(Mirror[rep[u][i]], "%d %d\n", u, master);
                fprintf(Master[master], " %d", rep[u][i]);
            }
        }
        fprintf(Master[master], "\n");
    }

    for (i = 0; i < nparts; i++) {
        fclose(G[i]);
        fclose(Master[i]);
        fclose(Mirror[i]);
    }

    printf("%s part finished!\n", partStrategy);
}

void nodecsv2txt(int vertexSize) {
    float v;
    char sep;
    FILE *fpw;
    std::ifstream fpr;
    fpr.open("node-feat.csv");
    fpw = fopen("node_feat.txt", "w+");

    for (int i = 0; i < vertexSize; i++) {
        std::string line;
        getline(fpr, line);
        std::istringstream st(line);
        while (true) {
            st >> v >> sep;
            if (st.eof()) break;
            fprintf(fpw, "%.8f ", v);
        }
        fprintf(fpw, "%.8f\n", v);
    }
//    fclose(fpr);
    fclose(fpw);
}

void reduce_graph(char* filePath, int workerPerNode = 4) {
    int *active;
    active = new int [maxn];

    vector<int> *rep;
    rep = new vector<int> [maxn];
    int i, u, v, j, assign;
    FILE *fp;

    fp = fopen("vertex_active.txt", "r");
    for (i = 0; i < maxn; i++) {
        fscanf(fp, "%d", &active[i]);
    }
    fclose(fp);

    char path[100];
    int *convert;

    for (i = 0; i < nparts; i++) {
        for (j = 0; j < maxn; j++) has[i][j] = false;
    }

    FILE *GI[nparts], *GO[nparts], *Master[nparts], *Mirror[nparts];

    for (i = 0; i < nparts; i++) {
        sprintf(path, "%s_/G.%d", filePath, i);
        GI[i] = fopen(path, "r");
        sprintf(path, "%s/G.%d", filePath, i);
        GO[i] = fopen(path, "w+");
        sprintf(path, "%s/Master.%d", filePath, i);
        Master[i] = fopen(path, "w+");
        sprintf(path, "%s/Mirror.%d", filePath, i);
        Mirror[i] = fopen(path, "w+");
    }

    int **has_node;
    has_node = new int*[nparts / workerPerNode];
    for (i = 0; i < nparts / workerPerNode; i++) {
        has_node[i] = new int[maxn];
        for (j = 0; j < maxn; j++) has_node[i][j] = 0;
    }

    int edge_count = 0;

    for (i = 0; i < nparts; i++) {
        while (~fscanf(GI[i], "%d%d", &u, &v)) {
            if (active[u] == 0 || active[v] == 0) continue;
//            printf("active u:%d, v:%d\n", u, v); fflush(stdout);
            if (!has[i][u]) {
                has[i][u] = true;
                rep[u].push_back(i);
                has_node[i / workerPerNode][u]++;
            }
//            printf("ok 1\n", u, v); fflush(stdout);
            if (!has[i][v]) {
                has[i][v] = true;
                rep[v].push_back(i);
                has_node[i / workerPerNode][v]++;
            }
//            printf("ok 2\n", u, v); fflush(stdout);

            fprintf(GO[i], "%d %d\n", u, v);
//            printf("ok 3\n", u, v); fflush(stdout);
        }
    }
    int nodes = nparts / workerPerNode;
    for (u = 0; u < maxn; u++) {
        int len = rep[u].size();
        if (len <= 1) continue;

        int max_node_count = 0;
        for (i = 0; i < nodes; i++) {
            if (max_node_count < has_node[i][u]) {
                max_node_count = has_node[i][u];
            }
        }

        int m_id = u % len;
        int master = rep[u][m_id];
        while (has_node[master / workerPerNode][u] != max_node_count) {
            m_id = (m_id + rand()) % len;
            master = rep[u][m_id];
        }

        fprintf(Master[master], "%d", u);
        for (i = 0; i < len; i++) {
            if (i != m_id) {
                fprintf(Mirror[rep[u][i]], "%d %d\n", u, master);
                fprintf(Master[master], " %d", rep[u][i]);
            }
        }
        fprintf(Master[master], "\n");
    }

    for (i = 0; i < nparts; i++) {
        fclose(GI[i]);
        fclose(GO[i]);
        fclose(Master[i]);
        fclose(Mirror[i]);
    }
}


int main(int argc,  char **argv)
{
	init("friend");
//    init("products");
//    init("papers100M-bin");
//    init("reddit");

//	init("products");
//	transfer2METIS_input();
//	transfer2DRONE(false);
//    METIS2DRONE();
	convertDNE2PSHEP("DNE");
//    nodecsv2txt(2449029);
//    generateInOutDegree();
//    reduce_graph(argv[1], 4);
}