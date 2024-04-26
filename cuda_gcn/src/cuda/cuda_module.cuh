#ifndef CUDA_MODULE_CUH
#define CUDA_MODULE_CUH

#include "cuda_kernel.cuh"
#include "../common/utils.h"
#include "cuda_cache.cuh"
#include "cuda_variable.cuh"
#include "string"
#include <cublas_v2.h>
#include <cusparse_v2.h>
//extern "C" {
#include "graph.h"
//}
//#include "../seq/gcn.h"
#include "cuda_communicate.cuh"
#include "seq.h"

#define CHECK_CUSPARSE(status)                                                   \
{                                                                               \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

using std::vector;
using std::pair;
using std::string;

class CUDAModule {
public:
    virtual void forward(bool) = 0;
    virtual void backward() = 0;
    virtual ~CUDAModule() {};
	string ModuleName;
};

class CUDAMatmul: public CUDAModule {
    CUDAVariable *a, *b, *c;
    int *Mirror2Worker, workerID;
    int m, n, p;
	cublasHandle_t handle;
public:
    CUDAMatmul(CUDAVariable *a, CUDAVariable *b, CUDAVariable *c, int m, int n, int p, int* Mirror2Worker, string ModuleName, int workerID);
    ~CUDAMatmul() {}
    void forward(bool);
    void backward();
};

class CUDASparseMatmul: public CUDAModule {
    CUDAVariable *a, *b, *c;
    CUDASparseIndex *sp;
    int m, n, p;
public:
    CUDASparseMatmul(CUDAVariable *a, CUDAVariable *b, CUDAVariable *c, CUDASparseIndex *sp, int m, int n, int p);
    ~CUDASparseMatmul() {}
    void forward(bool);
    void backward();
};

class CUDAGraphSum: public CUDAModule {
    CUDAVariable *in, *out;
    Graph *graph;
	Comm *comm;
    Buffer* buffer;
	Degree* degree;
	bool* active;
    CacheBuffer* cacheBuffer;
    int dim;
    int workerId;
    int workerNum;
	int train_step;
	size_t bufferSize;
	float* sparseBuffer;
	cusparseHandle_t handle = NULL;
	cusparseSpMatDescr_t matA, matAT;
	int *cscColPtr, *cscRowInd;
	float *cscValues;
	cusparseDnMatDescr_t inDataMat, outDataMat, inGradMat, outGradMat;
    int layer;
public:
    CUDAGraphSum(CUDAVariable *in, CUDAVariable *out, Graph* graph, Comm* comm, Degree* degree, Buffer* buffer, int dim,
                 int workerId, int workerNum, string Name, int layer);
    ~CUDAGraphSum();
    void forward(bool);
    void backward();
};

class CUDACrossEntropyLoss: public CUDAModule {
    CUDAVariable *logits;
    int *truth;
    float *loss;
    float *logits_max;
    int num_classes;
    
    float *d_loss;
    int *d_count;
public:
    CUDACrossEntropyLoss(CUDAVariable *logits, int *truth, float *loss, int num_classes, string ModuleName);
    ~CUDACrossEntropyLoss();
    void forward(bool);
    void backward();
};

class CUDAReLU: public CUDAModule {
    CUDAVariable *in;
    bool *mask;
public:
    CUDAReLU(CUDAVariable *in, string ModuleName);
    ~CUDAReLU();
    void forward(bool);
    void backward();
};

class CUDADropout: public CUDAModule {
    CUDAVariable *in;
    int *mask;
    float p;
public:
    CUDADropout(CUDAVariable *in, float p, string ModuleName);
    ~CUDADropout();
    void forward(bool);
    void backward();
};

class CUDAAdamVariable {
public:
    float *data, *grad, *m, *v;
    int size;
    bool decay;

    CUDAAdamVariable(CUDAVariable*, bool);
    ~CUDAAdamVariable();
};

class CUDAAdam {
    AdamParams params;
    int step_count;
    vector<CUDAAdamVariable*> vars;
public:
    CUDAAdam() {}
    CUDAAdam(vector<pair<CUDAVariable*, bool>> vars, AdamParams params);
    ~CUDAAdam();
    void step();
};

#endif
