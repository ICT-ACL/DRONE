#ifndef UTILS_H

typedef enum {
    FULL_BATCH,
    SEMI_BATCH
} BatchType;

const BatchType batch_type = SEMI_BATCH;

const bool compress = true;
const bool time_debug = true;
const int max_grid_size = 102400;
const bool sc = false;

extern long long master2mirror_forward_total[2], mirror2master_forward_total[2], master2mirror, mirror2master,
            master2mirror_backward_total[2], mirror2master_backward_total[2];

extern int current_epoch;
extern float mean_accuracy;
extern bool update_threshold;

//__global__
//void cuda_set_float_global(float *a, int len, float val) {
//	int id = threadIdx.x + blockIdx.x * blockDim.x;
//	int stride = gridDim.x * blockDim.x;
//
//	while (id < len) {
//		a[id] = val;
//		id += stride;
//	}
//}
//
//void device_set_float(float *a, int len, float val) {
//	cuda_set_int_global<<<std::min(max_grid_size, len / 32 + 1), 32>>>(a, len, val);
//}

#define UTILS_H
#endif