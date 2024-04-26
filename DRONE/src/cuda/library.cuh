#ifndef DRONE_CUDA_LIBRARY_CUH
#define DRONE_CUDA_LIBRARY_CUH

//__global__ void hello();

void print_from_cpu();

typedef struct Graph {
    int* a, len;
//    void NewGraph(int len);
//    void DelectGraph();
//    int sum(int bdim, int tdim);
} Graph;

void NewGraph(Graph* g, int len);

void DelectGraph(Graph* g);

int sum(Graph* g, int bdim, int tdim);

#endif //DRONE_CUDA_LIBRARY_CUH
