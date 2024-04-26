#include "cuda_compress.cuh"

// using shared memory
// stride = 1 << k; while dim >> k == 1
// assert stride == 32
// shared memory: float * stride * 2
__global__
void findMinMax(float *data, int maxn, int dim, const int stride, float *maxmin) {
    extern __shared__ float partial_max[];
    float *partial_min = partial_max + stride;

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    while (bid < maxn) {
        int gid = bid * dim + tid;

//		if (bid == 1234) {
//			printf("bid:%d, tid:%d, val:%.8	f\n", bid, tid, data[gid]);
//		}

        partial_max[tid] = data[gid];
        partial_min[tid] = data[gid];

        if (tid < dim - stride) {
            partial_max[tid] = fmaxf(partial_max[tid], data[gid + stride]);
            partial_min[tid] = fminf(partial_min[tid], data[gid + stride]);
        }
        int stride_ = stride >> 1;

        __syncthreads();
        if (tid < stride_) {
            volatile float *vol_max = partial_max;
            volatile float *vol_min = partial_min;
            float register val_t;
            while (stride_ > 0) {
                val_t = vol_max[tid + stride_];
                if (vol_max[tid] < val_t) vol_max[tid] = val_t;
                val_t = vol_min[tid + stride_];
                if (vol_min[tid] > val_t) vol_min[tid] = val_t;
                stride_ >>= 1;
            }
        }
        if (tid == 0) {
            maxmin[bid * 2] = partial_max[0];
            maxmin[bid * 2 + 1] = partial_min[0];

//			if (bid == 1234) {
//				printf("max:%.8f, min:%.8f\n -------------- \n", partial_max[0], partial_min[0]);
//			}
        }
        bid += gridDim.x;
    }
}

__global__
void convert(float *in, uint8_t* out, int maxn, int dim, float *maxmin) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    while (bid < maxn) {
        float max_ = maxmin[2 * bid];
        float min_ = maxmin[2 * bid + 1];

        out[bid * dim + tid] = uint8_t((in[bid * dim + tid] - min_) / (max_ - min_) * 255 + 0.5);
//        if (bid == 467903) {
//            float ori = in[bid * dim + tid];
//            float compressed = min_ + (max_ - min_) * out[bid * dim + tid] / 255;
//
//            printf("send bid:%d, tid:%d, ori:%.8f, compressed:%.8f, diff:%.8f\n", bid, tid, ori, compressed, ori - compressed);
//        }
//        if (tid == 0 && bid == 467903) printf("--------------\n");

        bid += gridDim.x;
    }
}

__global__
void de_convert(float *out, uint8_t* in, int maxn, int dim, float *maxmin) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    while (bid < maxn) {
        float max_ = maxmin[2 * bid];
        float min_ = maxmin[2 * bid + 1];

        out[bid * dim + tid] = min_ + (max_ - min_) * in[bid * dim + tid] / 255;

        bid += gridDim.x;
    }
}

void compressFloat2Uint8(float *in, uint8_t* out, int maxn, int dim, float *maxminSend) {
    int stride;
    for (int k = 0; k < 32; k++) {
        if ((dim - 1) >> k == 1) {
            stride = 1 << k;
            break;
        }
    }
//    printf("---------stride: %d, dim:%d\n-----", stride, dim);
    findMinMax<<<min(102400, maxn), stride, 2 * sizeof(float) * stride>>>(in, maxn, dim, stride, maxminSend);
    convert<<<min(102400, maxn), dim>>>(in, out, maxn, dim, maxminSend);
    cudaDeviceSynchronize();
}

__global__
void apply_sum_compressed(float* data, int* recvID, float* maxmin, uint8_t* recvIn, int* Global2Local, int recvLen, int feature_size) {
    int id = blockIdx.x;
    int stride = gridDim.x;
    int tid = threadIdx.x;

    while (id < recvLen) {
        float max_ = maxmin[2 * id];
        float min_ = maxmin[2 * id + 1];

        int localID = Global2Local[recvID[id]];
        float val = min_ + (max_ - min_) * recvIn[id * feature_size + tid] / 255;
        atomicAdd(data + localID * feature_size + tid, val);
        id += stride;
    }
}

__global__
void apply_assign_compressed(float* data, int* recvID, float* maxmin, uint8_t* recvIn, int* Global2Local, int recvLen, int feature_size) {
    int id = blockIdx.x;
    int stride = gridDim.x;
    int tid = threadIdx.x;

    while (id < recvLen) {
        float max_ = maxmin[2 * id];
        float min_ = maxmin[2 * id + 1];

        int localID = Global2Local[recvID[id]];
        float val = min_ + (max_ - min_) * recvIn[id * feature_size + tid] / 255;
        data[localID * feature_size + tid] = val;
        id += stride;
    }
}