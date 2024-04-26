#include "seq.h"
#include <immintrin.h>

using namespace std;

// sparse

void SparseIndex::print() {
    std::cout << "---sparse index info--" << endl;

    std::cout << "indptr: ";
    for (auto i: indptr) {
        cout << i << " ";
    }
    cout << endl;

    cout << "indices: ";
    for (auto i: indices) {
        cout << i << " ";
    }
    cout << endl;
}

// variable

Variable::Variable(int size, bool requires_grad):
        data(size), grad(requires_grad ? size : 0) {}

void Variable::glorot(int in_size, int out_size) {
    float range = sqrtf(6.0f / (in_size + out_size));

    for(int i = 0; i < data.size(); i++) {
        const float rand = float(RAND()) / MY_RAND_MAX - 0.5;
        data[i] = rand * range * 2;
    }
}

void Variable::zero() {
    for(int i = 0; i < data.size(); i++)
        data[i] = 0;
}

void Variable::zero_grad() {
    for(int i = 0; i < grad.size(); i++)
        grad[i] = 0;
}

void Variable::print(int col) {
    int count = 0;
    for(float x: data) {
        printf("%.4f ", x);
        count++;
        if(count % col == 0) printf("\n");
    }
}

float Variable::grad_norm() {
    float norm = 0;
    for(float x: grad) norm += x * x;
    return sqrtf(norm);
}


// optimize
AdamParams AdamParams::get_default() {
    return {0.001, 0.9, 0.999, 1e-8, 0.0};
}

AdamVariable::AdamVariable(Variable *var, bool decay):
        data(&var->data), grad(&var->grad), m(var->data.size(), 0.0), v(var->data.size(), 0.0), decay(decay) {}

int AdamVariable::size() {
    return data->size();
}

Adam::Adam(std::vector<std::pair<Variable*, bool>> vars, AdamParams params) {
    step_count = 0;
    this->params = params;
    for (auto v: vars)
        this->vars.emplace_back(v.first, v.second);
}

void Adam::step() {
    step_count++;
    float step_size = params.lr * sqrtf(1 - powf(params.beta2, step_count)) / (1 - powf(params.beta1, step_count));

    for (auto &var: vars) {
        for (int i = 0; i < var.size(); i++) {
            float grad = (*var.grad)[i];
            if (var.decay) grad += params.weight_decay * (*var.data)[i];
            var.m[i] = params.beta1 * var.m[i] + (1.0 - params.beta1) * grad;
            var.v[i] = params.beta2 * var.v[i] + (1.0 - params.beta2) * grad * grad;
            (*var.data)[i] -= step_size * var.m[i] / (sqrtf(var.v[i]) + params.eps);
        }
    }
}

GCNParams GCNParams::get_default() {
    return {2708, 1000, 1433, 64, 7, 0.5, 0.01, 5e-4, 5000, 0};
}

void GCNParams::Describe() {
    std::cout << "WorkerId:" << workerId
              << "globalVertexSize:" << globalVertexSize
              << "\tlocalVertexSize:" << localVertexSize
              << "\tinput_dim:" << input_dim
              << "\thidden_dim:" << hidden_dim
              << "\toutput_dim:" << output_dim << std::endl
              << "dropout:" << dropout
              << "\tlearning_rate:" << learning_rate
              << "\tweight_decay:" << weight_decay
              << "\tepochs:" << epochs
              << "\tearly_stopping:" << early_stopping
              << std::endl;
}

//rand
uint64_t rand_state[2];
void init_rand_state() {
    srand((unsigned)time(NULL));
    int x = 0, y = 0;
    while (x == 0 || y== 0) {
        x = rand();
        y = rand();
    }
    rand_state[0] = x;
    rand_state[1] = y;
}

uint32_t xorshift128plus(uint64_t* state) {
    uint64_t t = state[0];
    uint64_t const s = state[1];
    assert(t && s);
    state[0] = s;
    t ^= t << 23;		// a
    t ^= t >> 17;		// b
    t ^= s ^ (s >> 26);	// c
    state[1] = t;
    uint32_t res = (t + s) & 0x7fffffff;
    return res;
}
