#include <sstream>
#include <algorithm>
#include <cmath>
#include "utils.h"
#include "parser.h"

using namespace std;
Parser::Parser(GCNParams *gcnParams, GCNData *gcnData, std::string graph_name) {
//    string root = "data/";
    if (sc) {
        root = "/BIGDATA1/acict_zguan_1/zpltys/products/";
        this->split_file.open(root + "split.txt");
        this->feature_file.open(root + "node-feat.csv");
        this->label_file.open(root + "node-label.csv");
    } else {
        root = "/slurm/zhangshuai/GNN_Dataset/products/";
        this->split_file.open(root + "split/sales_ranking/split.txt");
        this->feature_file.open(root + "raw/node-feat.csv");
        this->label_file.open(root + "raw/node-label.csv");
    }
    this->gcnParams = gcnParams;
    this->gcnData = gcnData;
}

__global__ void setDegree(int *localDegree, int *globalDegree, int *Local2Global, int localVertexSize) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	while (id < localVertexSize) {
		int globalID = Local2Global[id];
		localDegree[id] = globalDegree[globalID];

		id += stride;
	}
}

void loadInOutDegree(Graph *g, int* degree_D, int *globalOutDegree) {
	cudaSetDevice(g -> GID);

	int globalVertexSize = *g->globalVertexSize;
	int localVertexSize = *g->localVertexSize;
	int *globalOutDegreeDevice;
//	cudaMalloc(&degree_D, sizeof(int) * localVertexSize);
	CUDA_CHECK(cudaMalloc(&globalOutDegreeDevice, sizeof(int) * globalVertexSize));
	cudaMemcpy(globalOutDegreeDevice, globalOutDegree, sizeof(int) * globalVertexSize, cudaMemcpyHostToDevice);

	setDegree<<<g->gridSize, g->blockSize>>>(degree_D, globalOutDegreeDevice, g->Local2Global, localVertexSize);

	cudaFree(globalOutDegreeDevice);
}

__global__
void test_variable(int* array) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index == 0) printf("test ok! --- array[0]:%d\n", array[0]);
}

void Parser::parseGraph(int globalVertexSize, int workerId, int workerNum, char* partStrategy) {
	// load G.x
    char filename[100];
    char filePre[100];

    if (sc) sprintf(filePre, "%s/product_%d_%s", root.c_str(), workerNum, partStrategy);
    else sprintf(filePre, "%sraw/product_%d_%s", root.c_str(), workerNum, partStrategy);

    sprintf(filename, "%s/G.%d", filePre, workerId);
    printf("filename: %s\n", filename);
    FILE* fp = fopen(filename, "r");
    vector<int> x, y;
    int u, v;
    while (~fscanf(fp, "%d %d", &u, &v)) {
//        printf("u: %d, v:%d\n", u, v);
        x.push_back(u);
        y.push_back(v);
    }
    fclose(fp);

    gcnData->comm = new Comm();
    gcnData->graph = build_graph(globalVertexSize, int(x.size()), x.data(), y.data(), workerId, workerNum, gcnData->comm);
//    printf("end build_graph\n");

	// load Master file
    std::ifstream fstream;
    sprintf(filename, "%s/Master.%d", filePre, workerId);
//    printf("filename: %s\n", filename);
//    fp = fopen(filename, "r");
    fstream.open(filename);
    vector<int> masterVertex, mirrorNumber, mirrorWorkers;

    std::string line;
    while (true) {
        getline(fstream, line);
        if (fstream.eof()) break;

        std::istringstream ss(line);
        ss >> u;
//        printf("u: %d\n", u);
        masterVertex.push_back(u);
		int num = 0;

        while (true) {
            ss >> v;
            if (ss.fail()) break;
//            printf("v: %d\n", v);
            num += 1;
            mirrorWorkers.push_back(v);
//            if (v < 0 || v > 4) printf("u: %d, v:%d\n", u, v);
        }
        mirrorNumber.push_back(num);
    }
    printf("mirrorNumber.size:%d\n", mirrorNumber.size());
    fstream.close();
//    printf("mirrorWorkers.size:%d\n", mirrorWorkers.size());
    addMasterRoute(gcnData->graph, masterVertex.data(), mirrorNumber.data(),
                   mirrorWorkers.data(), masterVertex.size(), mirrorWorkers.size());

    printf("load master ok, masterVertex.size(): %d, mirrorNumber.size(): %d!\n", masterVertex.size(), mirrorNumber.size());

	//load Mirror
    sprintf(filename, "%s/Mirror.%d", filePre, workerId);
    printf("mirror filename: %s\n", filename);
    fp = fopen(filename, "r");
    vector<int> mirrorVertex, masterWorker;
    while (~fscanf(fp, "%d %d", &u, &v)) {
        mirrorVertex.push_back(u);
        masterWorker.push_back(v);
    }
    printf("start addMirrorRoute\n");
	addMirrorRoute(gcnData->graph, mirrorVertex.data(), masterWorker.data(), mirrorVertex.size());
    fclose(fp);

    printf("load mirror ok, mirrorVertex.size():%d!\n", mirrorVertex.size());

	// load in degree
    gcnData->degree = new Degree();
	cudaMalloc(&gcnData->degree->inDegree, sizeof(int) * (*gcnData->graph->localVertexSize));
	cudaMalloc(&gcnData->degree->outDegree, sizeof(int) * (*gcnData->graph->localVertexSize));

	sprintf(filename, "%s/raw/in_degree.txt", root.c_str());
	fp = fopen(filename, "r");
	int* degree = new int[*gcnData->graph->globalVertexSize];
	for (int i = 0; i < *gcnData->graph->globalVertexSize; i++) {
		fscanf(fp, "%d", &u);
		degree[i] = u;
	}
	loadInOutDegree(gcnData->graph, gcnData->degree->inDegree, degree);
	fclose(fp);

    printf("load in degree ok!\n");

	// load out degree
	sprintf(filename, "%s/raw/out_degree.txt", root.c_str());
	fp = fopen(filename, "r");
	for (int i = 0; i < *gcnData->graph->globalVertexSize; i++) {
		fscanf(fp, "%d", &u);
		degree[i] = u;
	}
	loadInOutDegree(gcnData->graph, gcnData->degree->outDegree, degree);
	fclose(fp);

    printf("load out degree ok!\n");

	delete []degree;

//	CUDA_CHECK(cudaDeviceSynchronize());
//	test_variable<<<10,20>>>(gcnData->degree->inDegree);
//	CUDA_CHECK(cudaDeviceSynchronize());

    this->gcnParams->globalVertexSize = globalVertexSize;
    this->gcnParams->localVertexSize = getLocalVertexSize(gcnData->graph);
    cudaMallocHost((void **) &global2local, sizeof(int) * globalVertexSize);
    CUDA_CHECK(cudaMemcpy(global2local, gcnData->graph->Global2Local, globalVertexSize * sizeof(int), cudaMemcpyDeviceToHost));

    cudaMallocHost((void **) &Mirror2Worker, sizeof(int) * (this->gcnParams->localVertexSize));
    CUDA_CHECK(cudaMemcpy(Mirror2Worker, gcnData->graph->Mirror2Worker, (this->gcnParams->localVertexSize) * sizeof(int), cudaMemcpyDeviceToHost));
}

Parser::~Parser(){
    cudaFreeHost(global2local);
    cudaFreeHost(Mirror2Worker);
}

//__global__
//void setKernel(int *local2kernel, int *kernel2local, int *Mirror2Worker, int localVertexSize, int *iter) {
//    int id = threadIdx.x + blockIdx.x * blockDim.x;
//    int stride = gridDim.x * blockDim.x;
//
//    for (; id < localVertexSize; id += stride) {
//        if (Mirror2Worker[id] == -1) {
//            local2kernel[id] = -1;
//            continue;
//        }
//        int kernelID = atomicAdd(iter, 1);
//        local2kernel[id] = kernelID;
//        kernel2local[kernelID] = id;
//    }
//}

//void Parser::parseKernel() {
//    Graph* graph = &this->gcnData->graph;
//    Kernel* kernel = &this->gcnData->kernel;
//    int localVertexSize = getLocalVertexSize(graph);
//    cudaMalloc(&kernel->local2kernel, sizeof(int) * localVertexSize);
//    cudaMalloc(&kernel->kernel2local, sizeof(int) * localVertexSize);
//    cudaMallocManaged(&kernel->kernelSize, sizeof(int));
//    setKernel<<<graph->gridSize, graph->blockSize>>>(kernel->local2kernel, kernel->kernel2local, graph->Mirror2Worker, localVertexSize, kernel->kernelSize);
//}

bool Parser::isValidInput() {
    return split_file.is_open() && feature_file.is_open() && label_file.is_open();
}

void Parser::parseNode() {
    auto &feature_val = this->gcnData->feature_value;
    auto &labels = this->gcnData->label;

    int max_label = 0;
    float v;
	char sep;
    for (int i = 0; i < gcnParams->globalVertexSize; i++) {
        int localId = global2local[i];

        std::string line;
        getline(feature_file, line);
        if (localId != -1) {
            if (feature_val.empty()) {
                std::istringstream st(line);
                int count = 0;
                while (true) {
                    char sep;
                    st >> v >> sep;
                    count += 1;
                    if (st.eof()) break;
                }
                gcnParams->input_dim = count;
                feature_val.resize(gcnParams->input_dim * gcnParams->localVertexSize);
                labels.resize(gcnParams->localVertexSize);
//				printf("count:%d, input_dim:%d\n", count, gcnParams->input_dim);
            }
//          int count = 0;
			std::istringstream ss(line);
//			cout << line << endl;
            for (int j = 0; j < gcnParams->input_dim; j++) {
                ss >> v >> sep;
//				printf("j: %d, v:%f\n", j, v);
                feature_val[localId * gcnParams->input_dim + j] = v;
            }
        }

		getline(label_file, line);
        int label;
        std::istringstream slabel(line);
        slabel >> label;
        if (localId != -1) labels[localId] = label;
        max_label = max(max_label, label);
    }
//    gcnParams->input_dim = max_len;
    gcnParams->output_dim = max_label + 1;
}

void Parser::parseSplit() {
    auto &split = this->gcnData->split;
    split.resize(gcnParams->localVertexSize);
    int s;

    for (int i = 0; i < gcnParams->globalVertexSize; i++) {
        split_file >> s;
        int localId = global2local[i];
        if (localId == -1) continue;
        split[localId] = s;
//        split[localId] = Mirror2Worker[localId] == -1 ? s : -1;
    }
}

void vprint(std::vector<int> v){
    for(int i:v)printf("%i ", i);
    printf("\n");
}

bool Parser::parse(int globalVertexSize, int workerId, int workerNum, char* partStrategy) {
    if (!isValidInput()) return false;
    std::cout << "Parse Graph Start." << endl;
    this->parseGraph(globalVertexSize, workerId, workerNum, partStrategy);
    std::cout << "Parse Graph Succeeded." << endl;
//    this->parseKernel();
//    std::cout << "Parse Kernel ID Succeeded." << endl;
    this->parseNode();
    std::cout << "Parse Node Succeeded." << endl;
    this->parseSplit();
    std::cout << "Parse Split Succeeded." << endl;

    gcnParams->Describe();
    return true;
}
