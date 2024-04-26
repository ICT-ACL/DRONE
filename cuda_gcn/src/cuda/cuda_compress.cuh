#ifndef CUDA_COMPRESS_CUH
#define CUDA_COMPRESS_CUH

#include "cuda_variable.cuh"
#include "cuda_runtime.h"
#include "cstdint"

__global__
void findMinMax(float *data, int maxn, int dim, const int stride, float *maxmin);

__global__
void convert(float *in, uint8_t* out, int maxn, int dim, float *maxmin);

__global__
void de_convert(float *out, uint8_t* in, int maxn, int dim, float *maxmin);

void compressFloat2Uint8(float *in, uint8_t* out, int maxn, int dim, float *maxminSend);

__global__
void apply_sum_compressed(float* data, int* recvID, float* maxmin, uint8_t* recvIn, int* Global2Local, int recvLen, int feature_size);

__global__
void apply_assign_compressed(float* data, int* recvID, float* maxmin, uint8_t* recvIn, int* Global2Local, int recvLen, int feature_size);

#endif