#ifndef CUDA_GCN_CUH
#define CUDA_GCN_CUH

//#include "../seq/gcn.h"
#include "cuda_variable.cuh"
#include "cuda_module.cuh"

#include "seq.h"

using std::vector;
using std::pair;

class CUDAGCN: CUDAModule {
public:
    vector<CUDAModule*> modules;
    vector<CUDAVariable> variables;
    CUDAVariable *input, *output;
    CUDAAdam *optimizer;
	Buffer *buffer;
    int *truth;
	int *Mirror2WorkerHost;
    float loss;
    float *d_l2_penalty;

    void set_input();
    void set_truth(int current_split);
    float get_accuracy();
    float get_l2_penalty();
    pair<float, float> train_epoch();
    pair<float, float> eval(int current_split);
    GCNData *data;

    void forward(bool training);
    void backward();
//public:
    GCNParams params;
    CUDAGCN(GCNParams params, GCNData *input_data, int workerId, int workerNum);
    CUDAGCN() {}
    ~CUDAGCN();
    void run();
    std::pair<float, float> run_epoch(int epoch);
	void save_paras(char* path);
	void load_paras(char* path);
//    void forward(bool);
//    void backward();
};

#endif
