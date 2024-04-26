#ifndef CUDA_VARIABLE_CUH
#define CUDA_VARIABLE_CUH

#include "cuda_kernel.cuh"
#include "seq.h"
//#include "cuda_communicate.cuh"

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

struct CUDAVariable {
public:
    float *data, *grad;
    bool requires_grad, is_weight;
    int size;
    CUDAVariable(int size, bool requires_grad=true, bool is_weight=false);
    ~CUDAVariable();

    void glorot(int in_size, int out_size);
    void zero();
    void zero_grad();
    void print(int col);
    float grad_norm();
};

struct CUDASparseIndex {
public:
    int *indices, *indptr;
	int *in_degree;
    int indices_size, indptr_size;

    CUDASparseIndex(): indices(nullptr), indptr(nullptr), indices_size(0), indptr_size(0),in_degree(nullptr){}
    CUDASparseIndex(const SparseIndex &sp);
    ~CUDASparseIndex();
};

#endif
