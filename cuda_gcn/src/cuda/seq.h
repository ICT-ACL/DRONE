#ifndef SEQ_H
#include <string>
#include <sstream>
#include "../common/utils.h"
#include <iostream>
#include <vector>
#include <utility>
//extern "C" {
#include "graph.h"
//}

//#include <cstdlib>
#include <cstdint>
#include <assert.h>

// sparse
class SparseIndex {
public:
    std::vector<int> indices;
    std::vector<int> indptr;
    std::vector<int> in_degree;
    void print();
};


// variable

struct Variable {
    std::vector<float> data, grad;
    Variable(int size, bool requires_grad=true);
    void glorot(int in_size, int out_size);
    void zero();
    void zero_grad();
    void print(int col=0x7fffffff);
    float grad_norm();
};

// optimize
struct AdamParams {
    float lr, beta1, beta2, eps, weight_decay;
    static AdamParams get_default();
};

struct AdamVariable {
    std::vector<float> *data, *grad, m, v;
    bool decay;
public:
    int size();
    AdamVariable(Variable*, bool);
};

class Adam {
    AdamParams params;
    int step_count;
    std::vector<AdamVariable> vars;
public:
    Adam() {}
    Adam(std::vector<std::pair<Variable*, bool>> vars, AdamParams params);
    void step();
};

// gcn
struct GCNParams {
    int globalVertexSize, localVertexSize;
    int input_dim, hidden_dim, output_dim;
    float dropout, learning_rate, weight_decay;
    int epochs, early_stopping;
    int workerId, workerNum;
    static GCNParams get_default();
    void Describe();
};

struct Degree {
    int *inDegree, *outDegree;
};


class GCNData {
public:
//    SparseIndex graph;
    Graph* graph;
    Degree* degree;
//    Kernel kernel;
    Comm* comm;
    std::vector<int> split;
    std::vector<int> label;
    std::vector<float> feature_value;
};

// rand
#define MY_RAND_MAX 0x7fffffff

void init_rand_state();
uint32_t xorshift128plus(uint64_t* state);
extern uint64_t rand_state[2];
#define RAND() xorshift128plus(&rand_state[0])

#define SEQ_H
#endif
