extern "C" {
#include "common.h"
}
#include "stdio.h"

__global__ void count(int *active, int localVertexSize, int *res) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    while (id < localVertexSize) {
        if (active[id] != 0) {
            atomicAdd(res, 1);
        }
        id += stride;
    }
}

int countActive(int *active, int localVertexSize, int gridSize, int blockSize) {
    int *res_h, *res_d;
    cudaMalloc(&res_d, sizeof(int));
    CUDACHECK(cudaMallocHost(&res_h, sizeof(int)));
    *res_h = 0;
    cudaMemcpy(res_d, res_h, sizeof(int), cudaMemcpyHostToDevice);
    count<<<gridSize, blockSize>>>(active, localVertexSize, res_d);
    CUDACHECK(cudaDeviceSynchronize());
    cudaMemcpy(res_h, res_d, sizeof(int), cudaMemcpyDeviceToHost);

    int res = *res_h;
    cudaFree(res_d);
    cudaFreeHost(res_h);
    return res;
}


__global__ void CUDA_FILL_INT(int *array, int val, int maxLen) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    while (id < maxLen) {
        array[id] = val;
        id += stride;
    }
}

__global__ void CUDA_FILL_float(float *array, float val, int maxLen) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    while (id < maxLen) {
        array[id] = val;
        id += stride;
    }
}


//__device__ inline void atomicAdd(float* address, float value) {
//    float old = value;
//    float new_old;
//    do {
//        new_old = atomicExch(address, 0.0f);
//        new_old += old;
//    } while ((old = atomicExch(address, new_old)) != 0.0f);
//}