//
// Created by zplty on 2023/4/7.
//
#include "mpi.h"

extern "C" {
#include "gcn.h"
}

#include <vector>
#include "parser.h"
#include <iostream>
#include <string>

#include "cuda/cuda_gcn.cuh"

using namespace std;

//typedef CUDAGCN* GCNHandle;


GCNHandle getHandle(int workerId, int workerNum, int globalVertexSize, char* partStrategy) {
	GCNParams params = GCNParams::get_default();
	GCNData* data = new GCNData;
	std::string input_name("test");
	Parser parser(&params, data, input_name);
    printf("start parser!\n");
	if (!parser.parse(globalVertexSize, workerId, workerNum, partStrategy)) {
		std::cerr << "Cannot read input: " << input_name << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << "RUNNING ON GPU!" << std::endl;
	CUDAGCN *handle = new CUDAGCN(params, data, workerId, workerNum);
    CUDACHECK(cudaDeviceSynchronize());
//    printf("create cudagcn\n");
//	return handle;
    return static_cast<GCNHandle>(handle);
}

void run(GCNHandle handle) {
    CUDAGCN *cuda_gcn = static_cast<CUDAGCN*>(handle);
	cuda_gcn->run();
}

void init_delay() {
    for (int i = 0; i < 2; i++) {
        master2mirror_forward_total[i] = 0;
        master2mirror_backward_total[i] = 0;
        mirror2master_forward_total[i] = 0;
        mirror2master_backward_total[i] = 0;
    }
}

int main(int argc, char** argv)
{
//    if (argc < 5) {
//        printf("error input!");
//        return -1;
//    }
    MPI_Init(&argc, &argv);
    int workerId, workerNum, globalVertexSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &workerId);
    MPI_Comm_size(MPI_COMM_WORLD, &workerNum);

    sscanf(argv[1], "%d", &globalVertexSize);


	GCNHandle handle = getHandle(workerId, workerNum, globalVertexSize, argv[2]);
//	run(an);
    CUDAGCN *cuda_gcn = static_cast<CUDAGCN*>(handle);
    cout << "Worker:" << workerId << " epochs:" << cuda_gcn->params.epochs << " early_stopping:" << cuda_gcn->params.early_stopping << endl;
//	cuda_gcn->load_paras("model.txt");
    init_delay();
    mean_accuracy = 0.0;
    for (int epoch = 1; epoch < cuda_gcn->params.epochs; epoch++) {
        cuda_gcn->run_epoch(epoch);
		if (batch_type == SEMI_BATCH) {
			if (epoch % 10 == 0) {
				printf("Worker %d average delay rate: mirror2master_forward_0: %.4f, master2mirror_forward_0: %.4f, mirror2master_backward_0: %.4f, master2mirror_backward_0: %.4f\n",
					   workerId,
                       1.0 / 9 * mirror2master_forward_total[0] / mirror2master,
					   1.0 / 9 * master2mirror_forward_total[0] / master2mirror,
					   1.0 / 9 * mirror2master_backward_total[0] / mirror2master,
					   1.0 / 9 * master2mirror_backward_total[0] / master2mirror);

				printf("Worker %d average delay rate: mirror2master_forward_1: %.4f, master2mirror_forward_1: %.4f, mirror2master_backward_1: %.4f, master2mirror_backward_1: %.4f\n",
					   workerId,
                       1.0 / 9 * mirror2master_forward_total[1] / mirror2master,
					   1.0 / 9 * master2mirror_forward_total[1] / master2mirror,
					   1.0 / 9 * mirror2master_backward_total[1] / mirror2master,
					   1.0 / 9 * master2mirror_backward_total[1] / master2mirror);
				init_delay();
			}

//			printf("Worker %d average delay rate: mirror2master_forward_0: %.4f, master2mirror_forward_0: %.4f, mirror2master_backward_0: %.4f, master2mirror_backward_0: %.4f\n",
//				   workerId,
//				   1.0 / mirror2master_forward_total[0] / mirror2master,
//				   1.0 / master2mirror_forward_total[0] / master2mirror,
//				   1.0 / mirror2master_backward_total[0] / mirror2master,
//				   1.0 / master2mirror_backward_total[0] / master2mirror);
//
//			printf("Worker %d average delay rate: mirror2master_forward_1: %.4f, master2mirror_forward_1: %.4f, mirror2master_backward_1: %.4f, master2mirror_backward_1: %.4f\n",
//				   workerId,
//				   1.0 / mirror2master_forward_total[1] / mirror2master,
//				   1.0 / master2mirror_forward_total[1] / master2mirror,
//				   1.0 / mirror2master_backward_total[1] / mirror2master,
//				   1.0 / master2mirror_backward_total[1] / master2mirror);
//			init_delay();

		}
//		if (epoch == 10) break;
    }
}