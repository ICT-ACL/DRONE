extern "C" {
    #include "library.cuh"
}
//#include "library.cuh"
#include <stdio.h>

__global__ void hello() {
    printf("hello cuda from %d-%d\n", blockIdx.x, threadIdx.x);
}

void print_from_cpu() {
    hello<<<5, 7>>>();
    cudaThreadSynchronize();
}

__global__ void _sum(int* ans, int* a, int len) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
//    printf("id:%d\n", id);
    int stride = gridDim.x * blockDim.x;
    int temp = 0;
    for (int i = id; i < len; i += stride) {
        temp += a[i];
    }
//    printf("id:%d, temp:%d\n", id, temp);
    ans[id] = temp;
}

void NewGraph(Graph* g, int len);

void DelectGraph(Graph* g);

int sum(Graph* g, int bdim, int tdim);

void NewGraph(Graph* g, int len) {
    g -> len = len;
    cudaMallocManaged(&(g -> a), sizeof(int) * len);
    for (int i = 0; i < len; i++) {
        g -> a[i] = i;
    }
}

void DelectGraph(Graph* g) {
    cudaFree(g -> a);
}

int sum(Graph* g, int bdim, int tdim) {
    int *ans;
    cudaMalloc(&ans, sizeof(int) * bdim * tdim);
    _sum<<<bdim, tdim>>>(ans, g -> a, g -> len);
    cudaThreadSynchronize();
    int *host_ans = new int[bdim * tdim];
    cudaMemcpy(host_ans, ans, sizeof(int) * bdim * tdim, cudaMemcpyDeviceToHost);
    int result = 0;
    for (int i = 0; i < bdim * tdim; i++) {
        result += host_ans[i];
    }
    return result;
}

