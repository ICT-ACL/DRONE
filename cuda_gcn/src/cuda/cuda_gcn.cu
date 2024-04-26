#include "timer.h"
#include <algorithm>
#include <thrust/transform.h>
#include "cuda_gcn.cuh"

using std::max;
using std::max_element;

CUDAGCN::CUDAGCN(GCNParams _params, GCNData *input_data, int workerId, int workerNum) {
//	int device;
//	cudaGetDevice(&device);
//	printf("now device:%d\n", device);

    cuda_init_random_state(MAX_THREAD_PER_BLOCK);

    this->params = _params;
    data = input_data;
    params.workerId = workerId;
    params.workerNum = workerNum;

	cudaMallocHost((void **) &Mirror2WorkerHost, sizeof(int) * params.localVertexSize);
	cudaMemcpy(Mirror2WorkerHost, data->graph->Mirror2Worker, sizeof(int) * params.localVertexSize, cudaMemcpyDeviceToHost);

    modules.reserve(8);
    variables.reserve(8);

//    printf("buffer_init start!\n");
    buffer = new Buffer();
//    printf("buffer_init ok!\n");

    // dropout
    variables.emplace_back(data->feature_value.size(), false, false);
    input = &variables.back();
//    printf("dropout ok!\n");
//    modules.push_back(new CUDADropout(input, params.dropout));
    
    //
    variables.emplace_back(params.localVertexSize * params.hidden_dim, true, false);
    CUDAVariable *layer1_var1 = &variables.back();
    variables.emplace_back(params.input_dim * params.hidden_dim, true, true);
    CUDAVariable *layer1_weight = &variables.back();
    layer1_weight->glorot(params.input_dim, params.hidden_dim);
//    modules.push_back(new CUDASparseMatmul(input, layer1_weight, layer1_var1, sp, params.num_nodes, params.input_dim, params.hidden_dim));
    modules.push_back(new CUDAMatmul(input, layer1_weight, layer1_var1, params.localVertexSize, params.input_dim, params.hidden_dim, data->graph->Mirror2Worker, string("matmul_layer1"), workerId));
//    printf("CUDAMatmul ok!\n");

    // graph sum
//    printf("CUDAGraphSum start!\n");
    variables.emplace_back(params.localVertexSize * params.hidden_dim, true, false);
//    printf("CUDAGraphSum create variable!\n");
//    printf("maxDim:%d\n", buffer->maxDim);
	buffer->maxDim = params.hidden_dim;
//    printf("hidden dim:%d\n", params.hidden_dim);
    CUDAVariable *layer1_var2 = &variables.back();
//    printf("???\n");
    modules.push_back(new CUDAGraphSum(layer1_var1, layer1_var2, data->graph, data->comm, data->degree, buffer,
                                       params.hidden_dim, params.workerId, params.workerNum, string("graphSum_layer1"), 0));
//    printf("CUDAGraphSum ok!\n");

    // ReLU
    modules.push_back(new CUDAReLU(layer1_var2, string("Relu")));

    // dropout
//    modules.push_back(new CUDADropout(layer1_var2, params.dropout));

    // matmul
    variables.emplace_back(params.localVertexSize * params.output_dim);
    CUDAVariable *layer2_var1 = &variables.back();
    variables.emplace_back(params.hidden_dim * params.output_dim, true, true);
    CUDAVariable *layer2_weight = &variables.back();
    layer2_weight->glorot(params.hidden_dim, params.output_dim);
    modules.push_back(new CUDAMatmul(layer1_var2, layer2_weight, layer2_var1, params.localVertexSize, params.hidden_dim, params.output_dim, data->graph->Mirror2Worker, string("matmul_layer2"), workerId));

    // graph sum
    variables.emplace_back(params.localVertexSize * params.output_dim);
	buffer->maxDim = max(params.output_dim, buffer->maxDim);
    output = &variables.back();
    modules.push_back(new CUDAGraphSum(layer2_var1, output, data->graph, data->comm, data->degree, buffer,
                                       params.output_dim, params.workerId, params.workerNum, string("graphSum_layer2"), 1));

    // cross entropy loss
    CUDA_CHECK(cudaMalloc((void**) &truth, params.localVertexSize * sizeof(int)));
    modules.push_back(new CUDACrossEntropyLoss(output, truth, &loss, params.output_dim, string("CrossEntropy")));

    // optimizer
    AdamParams adam_params = AdamParams::get_default();
    adam_params.lr = params.learning_rate;
    adam_params.weight_decay = params.weight_decay;
    optimizer = new CUDAAdam({{layer1_weight, true}, {layer2_weight, false}}, adam_params);

    CUDACHECK(cudaDeviceSynchronize());

    set_input();
    // other variable
    CUDA_CHECK(cudaMalloc((void**) &d_l2_penalty, variables[2].size * sizeof(float)));

	buffer_init(buffer, data->graph, params.workerId, params.workerNum);
    CUDACHECK(cudaDeviceSynchronize());
}

CUDAGCN::~CUDAGCN() {
//	printf("delete CUDAGCN\n");
    cuda_free_random_state();
    delete_buffer(buffer);
    for (auto &m : modules) delete m;
    delete optimizer;
	delete data;
    CUDA_CHECK(cudaFree(truth));
    CUDA_CHECK(cudaFree(d_l2_penalty));
}

__global__
void showVariable(float* val, int layer, int workerID, bool is_grad) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (is_grad) printf("grad --- workerID:%d, layer:%d, id:%d, value:%e\n", workerID, layer, id, val[id]);
	else printf("value --- workerID:%d, layer:%d, id:%d, value:%e\n", workerID, layer, id, val[id]);
}

void CUDAGCN::forward(bool training) {
    timer_start(TMR_TRAIN);
    timer_start(TMR_Bcast);
    weight_broadcast_value_nccl(&variables, data->comm);
	cudaDeviceSynchronize();
    float tmp = timer_stop(TMR_Bcast);
//    printf("Worker Id: %d weight_broadcast_value_nccl time:%.5f\n", this->params.workerId, tmp);

//	int layer = 1;
//	for (int i = 0; i < variables.size(); i++) {
//		if (variables[i].is_weight) {
//			showVariable<<<1,10>>>(variables[i].data, layer, params.workerId, false);
//			cudaDeviceSynchronize();
//			layer++;
//		}
//	}

    tmp = timer_stop(TMR_TRAIN);
//	printf("Worker Id: %d weight_broadcast_value_nccl time:%.5f, train time total:%.5f\n", this->params.workerId, tmp, timer_total(TMR_TRAIN));
    for (auto m: modules) {
//		timer_start(TMR_TMP);
		m->forward(true);
//        timer_stop(TMR_TMP);
//		printf("Worker %d %s forward time:%.5f\n", this->params.workerId, m->ModuleName.c_str(), timer_stop(TMR_TMP));
	}
}

void CUDAGCN::backward() {
    for (int i = modules.size() - 1; i >= 0; i--) {
//		timer_start(TMR_TMP);
		modules[i]->backward();
//        timer_stop(TMR_TMP);
//		printf("Worker %d %s backward time:%.5f\n", this->params.workerId, modules[i]->ModuleName.c_str(), timer_stop(TMR_TMP));
	}
    timer_start(TMR_TRAIN);
    timer_start(TMR_Bcast);
    weight_reduce_grad_nccl(&variables, data->comm);
	cudaDeviceSynchronize();
    float tmp = timer_stop(TMR_Bcast);
//    printf("Worker Id: %d weight_reduce_grad_nccl time:%.5f\n", this->params.workerId, tmp);

//	int layer = 1;
//	for (int i = 0; i < variables.size(); i++) {
//		if (variables[i].is_weight) {
//			showVariable<<<1,10>>>(variables[i].grad, layer, params.workerId, true);
//			cudaDeviceSynchronize();
//			layer++;
//		}
//	}
    tmp = timer_stop(TMR_TRAIN);
//	printf("Worker Id: %d weight_reduce_grad_nccl time:%.5f, train time total:%.5f\n", this->params.workerId, tmp, timer_total(TMR_TRAIN));
}

void CUDAGCN::save_paras(char* path) {
	FILE* fp = fopen(path, "w+");
	printf("create file %s ok!", path);
	for (int i = 0; i < variables.size(); i++) {
		if (variables[i].is_weight) {
			float *data;
			CUDA_CHECK(cudaMallocHost(&data, sizeof(float) * variables[i].size));
			CUDA_CHECK(cudaMemcpy(data, variables[i].data, sizeof(float) * variables[i].size, cudaMemcpyDeviceToHost));
			for (int j = 0; j < variables[i].size; j++) {
				fprintf(fp, "%f ", data[j]);
			}
			fprintf(fp, "\n");
			cudaFreeHost(data);
			printf("save para %d --- size:%d\n", i, variables[i].size);
		}
	}
	fclose(fp);
}
void CUDAGCN::load_paras(char* path) {
	FILE* fp = fopen(path, "r");
	for (int i = 0; i < variables.size(); i++) {
		if (variables[i].is_weight) {
			float *data;
			cudaMallocHost(&data, sizeof(float) * variables[i].size);
			for (int j = 0; j < variables[i].size; j++) {
				fscanf(fp, "%f", &data[j]);
			}
			cudaMemcpy(variables[i].data, data, sizeof(float) * variables[i].size, cudaMemcpyHostToDevice);
			cudaFreeHost(data);
			printf("load para %d --- size:%d\n", i, variables[i].size);
		}
	}
	fclose(fp);
}

void CUDAGCN::set_input() {
    CUDA_CHECK(cudaMemcpy(input->data, data->feature_value.data(), input->size * sizeof(float), cudaMemcpyHostToDevice));
}

void CUDAGCN::set_truth(int current_split) {
    int *d_data_split, *d_data_label;
    CUDA_CHECK(cudaMalloc((void**) &d_data_split, params.localVertexSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &d_data_label, params.localVertexSize * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data_split, data->split.data(), params.localVertexSize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data_label, data->label.data(), params.localVertexSize * sizeof(int), cudaMemcpyHostToDevice));
    dim3 block((params.localVertexSize-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    cuda_set_truth_kernel<<<block, thread_in_block>>>(truth, d_data_split, d_data_label, current_split, params.localVertexSize);
    CUDA_CHECK(cudaFree(d_data_split));
    CUDA_CHECK(cudaFree(d_data_label));
}

// TODO: reduction (using thrust?)
float CUDAGCN::get_accuracy() {
    int *cpu_truth = new int[params.localVertexSize];
    float *cpu_output = new float[output->size];
    CUDA_CHECK(cudaMemcpy(cpu_truth, truth, params.localVertexSize * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu_output, output->data, output->size * sizeof(float), cudaMemcpyDeviceToHost));

    int wrong = 0, total = 0;
    for(int i = 0; i < params.localVertexSize; i++) {
        if(cpu_truth[i] < 0 || Mirror2WorkerHost[i] != -1) continue;
        total++;
        float truth_logit = cpu_output[i * params.output_dim + cpu_truth[i]];
        for(int j = 0; j < params.output_dim; j++)
            if (cpu_output[i * params.output_dim + j] > truth_logit) {
                wrong++;
                break;
            }
    }
    delete[] cpu_truth;
    delete[] cpu_output;
    return float(total - wrong) / total;
}

struct square_functor{
    square_functor() {}
    __host__ __device__ float operator()(const float &x) const {
        return x * x;
    }
};
float CUDAGCN::get_l2_penalty() {
    int size = variables[2].size;
    thrust::device_ptr<float> l2_ptr(d_l2_penalty), var2_ptr(variables[2].data);
    thrust::transform(var2_ptr, var2_ptr + size, l2_ptr, square_functor());
    float l2 = thrust::reduce(l2_ptr, l2_ptr + size, (float)0.0, thrust::plus<float>());
    return params.weight_decay * l2 / 2;
}

void test_variables(CUDAGCN *cuda_gcn, int id, int feature_size) {
	assert(cuda_gcn->params.localVertexSize * feature_size == cuda_gcn -> variables[id].size);

	int* g2l;
	cudaMallocHost(&g2l, sizeof(int) * cuda_gcn->params.globalVertexSize);
	cudaMemcpy(g2l, cuda_gcn->data->graph->Global2Local, sizeof(int) * cuda_gcn->params.globalVertexSize, cudaMemcpyDeviceToHost);

	float* data;
	cudaMallocHost(&data, sizeof(float) * cuda_gcn->variables[id].size);
	cudaMemcpy(data, cuda_gcn->variables[id].data, sizeof(float) * cuda_gcn->variables[id].size, cudaMemcpyDeviceToHost);
	printf("---- variable %d ----\n", id);
	for (int i = 0; i < 10; i++) {
		int li = g2l[i];
		if (li == -1) continue;
		printf("%d:", i);
		for (int j = 0; j < 10; j++) {
			printf(" %f", data[li * feature_size + j]);
		}
		printf(" ...\n");
	}

	cudaFreeHost(data);
	cudaFreeHost(g2l);
}

pair<float, float> CUDAGCN::train_epoch() {
	cudaDeviceSynchronize();
    set_truth(1);
	cudaDeviceSynchronize();

    forward(true);

//	test_variables(this, 0, params.input_dim);
//	test_variables(this, 1, params.hidden_dim);
//	test_variables(this, 3, params.hidden_dim);

    float train_loss = loss + get_l2_penalty();
    float train_acc = get_accuracy();
//    for (int i = modules.size() - 1; i >= 0; i--)
//        modules[i]->backward();
    backward();
    optimizer->step();
	cudaDeviceSynchronize();
    return {train_loss, train_acc};
}

pair<float, float> CUDAGCN::eval(int current_split) {
//    set_input();
	cudaDeviceSynchronize();
    set_truth(current_split);
//    for (auto m: modules)
//        m->forward(false);
    forward(false);
    float test_loss = loss + get_l2_penalty();
    float test_acc = get_accuracy();
    return {test_loss, test_acc};
}

void CUDAGCN::run() {
//    int epoch = 1;
//
//    std::vector<float> loss_history;
//    for(; epoch <= params.epochs; epoch++) {
//        float train_loss, train_acc, val_loss, val_acc;
//        std::tie(train_loss, train_acc) = train_epoch();
//        float time_train = timer_total(TMR_TRAIN);
//        float time_comp = timer_total(TMR_COMP);
//        float time_comm = timer_total(TMR_COMM);
//        if (epoch % 10 == 0) {
//			std::tie(val_loss, val_acc) = eval(2);
//            printf("epoch=%d train_loss=%.5f train_acc=%.5f val_loss=%.5f val_acc=%.5f train_time=%.5f compute time=%.5f comm time=%.5f\n",
//                   epoch, train_loss, train_acc, val_loss, val_acc, time_train, time_comp, time_comm);
//        }
//        timer_clear(TMR_COMP);
//        timer_clear(TMR_COMM);
//        loss_history.push_back(val_loss);
//        if(params.early_stopping > 0 && epoch >= params.early_stopping) {
//            float recent_loss = 0.0;
//            for(int i = epoch - params.early_stopping; i < epoch; i++)
//                recent_loss += loss_history[i];
//            if (val_loss > recent_loss / params.early_stopping) {
//                printf("Early stopping...\n");
//                break;
//            }
//        }
//    }
//    printf("total training time=%.5f\n", timer_total(TMR_TRAIN));
//
//    float test_loss, test_acc;
//    timer_start(TMR_TEST);
//    std::tie(test_loss, test_acc) = eval(3);
//    printf("test_loss=%.5f test_acc=%.5f time=%.5f\n", test_loss, test_acc, timer_stop(TMR_TEST));
}

std::pair<float, float> CUDAGCN::run_epoch(int epoch) {
    current_epoch = epoch;
    float train_loss, train_acc, val_loss, val_acc;
//    timer_start(TMR_TRAIN);
    std::tie(train_loss, train_acc) = train_epoch();
    float time_train = timer_total(TMR_TRAIN);
    float time_comp = timer_total(TMR_COMP);
    float time_comm = timer_total(TMR_COMM_);
//	if (epoch == 10) exit(0);

    update_threshold = train_acc < (mean_accuracy - 0.001);
    mean_accuracy = 0.95f * mean_accuracy + 0.05f * train_acc;

    if (epoch % 10 == 0 || epoch < 10) {
        std::tie(val_loss, val_acc) = eval(2);
        printf("Worker %d, epoch=%d train_loss=%.5f train_acc=%.5f val_loss=%.5f val_acc=%.5f train_time=%.5f compute time=%.5f comm time=%.5f, mean_accuracy:%f\n",
               params.workerId, epoch, train_loss, train_acc, val_loss, val_acc, time_train, time_comp, time_comm, mean_accuracy);
    }
    timer_clear(TMR_COMP);
    timer_clear(TMR_COMM_);
    timer_clear(TMR_TRAIN);

    return std::make_pair(train_loss, train_acc);
}