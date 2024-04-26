#include "cuda_kernel.cuh"

curandState *devStates;

// matrix mult
__global__
void cuda_Matmul_forward_kernel(const float *a, const float *b, float *c, const uint m, const uint n, const uint p) {
	__shared__ float tileA[TILE_SIZE][TILE_SIZE];
	__shared__ float tileB[TILE_SIZE][TILE_SIZE];
	int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
//	printf("bx:%d, by:%d, tx:%d, ty:%d\n", bx, by, tx, ty);
	for (; bx <= (p - 1) / TILE_SIZE + 1; bx += TILE_SIZE) {
		for (; by <= (m - 1) / TILE_SIZE + 1; by += TILE_SIZE) {
			__syncthreads();

			int row = by * TILE_SIZE + ty;
			int col = bx * TILE_SIZE + tx;
			int range = (n - 1) / TILE_SIZE + 1;
			float res = 0;

#pragma unroll
			for (int i = 0; i < range; i++) {
				if (row < m && i * TILE_SIZE + tx < n)
					tileA[ty][tx] = a[row * n + i * TILE_SIZE + tx];
				else
					tileA[ty][tx] = 0;
				if (col < p && i * TILE_SIZE + ty < n)
					tileB[ty][tx] = b[(i * TILE_SIZE + ty) * p + col];
				else
					tileB[ty][tx] = 0;

				__syncthreads();
#pragma unroll
				for (int j = 0; j < TILE_SIZE; j++)
					res += tileA[ty][j] * tileB[j][tx];
				__syncthreads();
			}
			if (row < m && col < p)
				c[row * p + col] = res;
		}
	}
}

__global__
void cuda_Matmul_backward_A_kernel(float *a_grad, const float *b, const float *c_grad, const uint m, const uint n, const uint p) {
	__shared__ float tileB[TILE_SIZE][TILE_SIZE];
	__shared__ float tileCGrad[TILE_SIZE][TILE_SIZE];
	int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;

	for (; bx <= (n - 1) / TILE_SIZE + 1; bx += TILE_SIZE) {
		for (; by <= (m - 1) / TILE_SIZE + 1; by += TILE_SIZE) {
			__syncthreads();

			int row = by * TILE_SIZE + ty;
			int col = bx * TILE_SIZE + tx;
			int range = (p - 1) / TILE_SIZE + 1;
			float res = 0;
#pragma unroll
			for (int i = 0; i < range; i++) {
				if (row < m && i * TILE_SIZE + tx < p)
					tileCGrad[ty][tx] = c_grad[row * p + i * TILE_SIZE + tx];
				else
					tileCGrad[ty][tx] = 0;
				if (col < n && i * TILE_SIZE + ty < p)
					tileB[ty][tx] = b[col * p + i * TILE_SIZE + ty];
				else
					tileB[ty][tx] = 0;
				__syncthreads();

#pragma unroll
				for (int j = 0; j < TILE_SIZE; j++)
					res += tileCGrad[ty][j] * tileB[j][tx];
				__syncthreads();
			}
			if (row < m && col < n)
				a_grad[row * n + col] = res;
		}
	}
}

__global__
void cuda_Matmul_backward_B_kernel(float *b_grad, const float *a, const float *c_grad, const uint m, const uint n, const uint p, const int *Mirror2Worker) {
	__shared__ float tileA[TILE_SIZE][TILE_SIZE];
	__shared__ float tileCGrad[TILE_SIZE][TILE_SIZE];
	int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
	for (; bx <= (p - 1) / TILE_SIZE + 1; bx += TILE_SIZE) {
		for (; by <= (n - 1) / TILE_SIZE + 1; by += TILE_SIZE) {
			__syncthreads();

			int row = by * TILE_SIZE + ty;
			int col = bx * TILE_SIZE + tx;
			int range = (m - 1) / TILE_SIZE + 1;
			float res = 0;

#pragma unroll
			for (int i = 0; i < range; i++) {
				if (row < n && i * TILE_SIZE + tx < m)
					tileA[ty][tx] = a[(i * TILE_SIZE + tx) * n + row];
				else
					tileA[ty][tx] = 0;
				if (col < p && i * TILE_SIZE + ty < m && Mirror2Worker[i * TILE_SIZE + ty] == -1)
					tileCGrad[ty][tx] = c_grad[(i * TILE_SIZE + ty) * p + col];
				else
					tileCGrad[ty][tx] = 0;
				__syncthreads();

#pragma unroll
				for (int j = 0; j < TILE_SIZE; j++)
					res += tileA[ty][j] * tileCGrad[j][tx];
				__syncthreads();
			}
			if (row < n && col < p)
				b_grad[row * p + col] = res;
		}
	}
}


// sparse matmul
__global__
void cuda_SparseMatmul_forward_kernel(float *a_in, float *b_in, float *c_in, int *indptr, int *indices, int p) {
    int i = blockIdx.x;
    int k = threadIdx.x;

    #pragma unroll
    for (int jj = indptr[i]; jj < indptr[i + 1]; jj++) {
        int j = indices[jj];
        c_in[i * p + k] += a_in[jj] * b_in[j * p + k];
    }
}

__global__
void cuda_SparseMatmul_backward_kernel(float *a_in, float *b_in, float *c_in, int *indptr, int *indices, int p) {
    int i = blockIdx.x;
    int k = threadIdx.x;

    #pragma unroll
    for (int jj = indptr[i]; jj < indptr[i + 1]; jj++){
        int j = indices[jj];
        b_in[j * p + k] += c_in[i * p + k] * a_in[jj];
    }
}

__global__
void cuda_compute_csr_values(int *rowPtr, int *colInd, int *in_degree, int *outDegree, int localVertexSize, float *values) {
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;
	while (u < localVertexSize) {
		for (int i = rowPtr[u]; i < rowPtr[u + 1]; i++) {
			int v = colInd[i];
			values[i] = 1.0 / sqrtf(1.0 * outDegree[u] * in_degree[v]);
		}
		u += stride;
	}
}

// graph sum
__global__
void cuda_GraphSum_forward_kernel(float *d_in_data, float *d_out_data, int *d_indptr, int *d_indices, int dim, int numNodes) {
    int src = blockIdx.x;
    int j = threadIdx.x;

    int ptr_src_0 = d_indptr[src];
    int ptr_stc_1 = d_indptr[src + 1];

    #pragma unroll
    for (int i = ptr_src_0; i < ptr_stc_1; i++) {
        int dst = d_indices[i];
        float coef = 1.0 / sqrtf((ptr_stc_1 - ptr_src_0) * (d_indptr[dst + 1] - d_indptr[dst]));
        // This only works for undirected graphs. Should be out[dst] += coef * in[src]]
        d_out_data[src * dim + j] += coef * d_in_data[dst * dim + j];
    }
}

//__global__
//void test_variable(int* array) {
//	int index = threadIdx.x + blockIdx.x * blockDim.x;
//	int stride = blockDim.x * gridDim.x;
//
//	if (index == 0) printf("test ok! --- array[0]:%d\n", array[0]);
//}

__global__
void cuda_GraphSum_forward_active(float *d_in_data, float *d_out_date, int *d_indptr, int *d_indices, int *in_degree, int *outDegree, int dim, int localVertexSize, bool *active) {
	int index_src = blockIdx.x;
	int d_idx = threadIdx.x;
	int stride_src = gridDim.x;
	#pragma unroll
	for (int u = index_src; u < localVertexSize; u += stride_src) {
		for (int idx = d_indptr[u]; idx < d_indptr[u + 1]; idx++) {
			int v = d_indices[idx]; // edge u -> v
			float coef  = 1.0 / sqrtf(1.0 * outDegree[u] * in_degree[v]);
			d_out_date[v * dim + d_idx] += coef * d_in_data[u * dim + d_idx];
//			if (d_idx == 0) active[v] = true;
            active[v] = true;
		}
	}
}


//threadIdx:112, gridIdx:28
__global__
void cuda_GraphSum_forward(float *d_in_data, float *d_out_date, int *d_indptr, int *d_indices, int *in_degree, int *outDegree, int dim, int localVertexSize) {
    int index_src = blockIdx.x;
    int d_idx = threadIdx.x;
    int stride_src = gridDim.x;
    #pragma unroll
    for (int u = index_src; u < localVertexSize; u += stride_src) {
        for (int idx = d_indptr[u]; idx < d_indptr[u + 1]; idx++) {
            int v = d_indices[idx]; // edge u -> v
            float coef  = 1.0 / sqrtf(1.0 * outDegree[u] * in_degree[v]);
            d_out_date[v * dim + d_idx] += coef * d_in_data[u * dim + d_idx];
//			if(v == 0 && d_idx < 10) {
//				printf("edge: %d -> %d, coef:%f, d_out_date[%d]:%f\n", u, v, coef, v * dim + d_idx, d_out_date[v * dim + d_idx]);
//			}
        }
    }
}

__global__
void cuda_GraphSum_backward_active(float *d_in_grad, float *d_out_grad, int *d_indptr, int *d_indices, int *in_degree, int *outDegree, int dim, int localVertexSize, bool *active) {
	int index_src = blockIdx.x;
	int d_idx = threadIdx.x;
	int stride_src = gridDim.x;
	for (int u = index_src; u < localVertexSize; u += stride_src) {
		for (int idx = d_indptr[u]; idx < d_indptr[u + 1]; idx++) {
			int v = d_indices[idx]; // edge u -> v
			float coef  = 1.0 / sqrtf(1.0 * outDegree[u] * in_degree[v]);
			d_in_grad[u * dim + d_idx] += coef * d_out_grad[v * dim + d_idx];
		}
		if (d_idx == 0 && d_indptr[u + 1] - d_indptr[u] > 0) active[u] = true;
	}
}

__global__
void cuda_GraphSum_backward(float *d_in_grad, float *d_out_grad, int *d_indptr, int *d_indices, int *in_degree,
                                  int *outDegree, int dim, int localVertexSize) {
    int index_src = blockIdx.x;
    int d_idx = threadIdx.x;
    int stride_src = gridDim.x;
    for (int u = index_src; u < localVertexSize; u += stride_src) {
        for (int idx = d_indptr[u]; idx < d_indptr[u + 1]; idx++) {
            int v = d_indices[idx]; // edge u -> v
            float coef  = 1.0 / sqrtf(1.0 * outDegree[u] * in_degree[v]);
            d_in_grad[u * dim + d_idx] += coef * d_out_grad[v * dim + d_idx];
        }
    }
}

// cross entropy
__global__ 
void cuda_CrossEntropy_forward_A_kernel(float* logits_data, float* logits_grad, bool training, int num_classes, int* truth, int* count, float* thread_loss, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
//	if (i == 0) printf("cuda_CrossEntropy_forward_A_kernel --- i:%d, size:%d\n", i, size);
    if (i >= size) return;
    if (truth[i] < 0) {
        count[i] = 0;
        return;
    }
    float *logit = &logits_data[i * num_classes];
    float max_logit = -1e30, sum_exp = 0;
    #pragma unroll
    for (int j = 0; j < num_classes; j++)
        max_logit = fmax(max_logit, logit[j]);
    #pragma unroll
    for (int j = 0; j < num_classes; j++) {
        logit[j] -= max_logit;
//		if (i == 1645021) {
//			printf("original --- sum_exp add from %d --- %f\n", j, expf(logit[j]));
//		}
        sum_exp += expf(logit[j]);
    }
    if (training) {
        #pragma unroll
        for (int j = 0; j < num_classes; j++) {
            float prob = expf(logit[j]) / sum_exp;
            logits_grad[i * num_classes + j] = prob;
        }
        logits_grad[i * num_classes + truth[i]] -= 1.0;
//        if (i == 1645021) {
//            printf("original --- max_logit:%f, sum_exp:%f\n", max_logit, sum_exp);
//            for (int j = 0; j < num_classes; j++) {
//                printf("original --- logits_grad[%d]:%f\n", j, logits_grad[i * num_classes + j]);
//            }
//        }
    }
    count[i] = 1;
    thread_loss[i] = logf(sum_exp) - logit[truth[i]];
//    if (i == 1645021) {
//        printf("original --- thread_loss:%f\n", thread_loss[i]);
//    }
}

__global__
void cal_max(float *logits_value, float *logits_max, int size, int num_classes, int* truth, const int stride) {
    extern __shared__ float partial_max[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    while (bid < size) {
        if (truth[bid] < 0) {
            bid += gridDim.x;
            continue;
        }

        int id = bid * num_classes + tid;
        if (tid < num_classes - stride) {
            partial_max[tid] = fmaxf(logits_value[id], logits_value[id + stride]);
        } else partial_max[tid] = logits_value[id];
        int stride_ = stride >> 1;

        __syncthreads();
        if (tid < stride_) {
            volatile float *vol_max = partial_max;
            while (stride_ > 0) {
                vol_max[tid] = fmaxf(vol_max[tid], vol_max[tid + stride_]);
                stride_ >>= 1;
            }
        }
        if (tid == 0) logits_max[bid] = partial_max[0];
        bid += gridDim.x;
    }
}

__global__
void cuda_CrossEntropy_forward_A_kernel_(float* logits_data, float *logits_max, float* logits_grad, bool training, int num_classes, int* truth, int* count, float* thread_loss, int size) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ float sum_exp[1];

    while (bid < size) {
        if (truth[bid] < 0) {
            count[bid] = 0;
            bid += gridDim.x;
            continue;
        }

        int id = bid * num_classes + tid;
        float logits_value = logits_data[id] - logits_max[bid];
//        logits_data[id] -= logits_max[bid];
//        float exp_logit = expf(logits_data[id]);
        float exp_logit = expf(logits_value);
        sum_exp[0] = 0.0f;
        __syncthreads();
        atomicAdd(sum_exp, exp_logit);
//        if (bid == local_detect_id && tid >= 10 && tid < 15) {
//            printf("test --- sum_exp: %f add from %d --- %f, logits_value:%f, logits_max:%f\n", sum_exp[0], tid, exp_logit, logits_value, logits_max[bid]);
//        }
        __syncthreads();

        if (training) {
            logits_grad[id] = exp_logit / sum_exp[0];
            if (tid == truth[bid]) {
                logits_grad[id] -= 1.0;
            }
        }

//        if (bid == local_detect_id && tid >= 10 && tid < 15) {
//            if (tid == 0) printf("test --- max_logit:%f, sum_exp:%f\n", logits_max[bid], sum_exp);
//            printf("test --- logits_grad[%d]:%f\n", tid, logits_grad[id]);
//        }
        if (tid == truth[bid]) {
            count[bid] = 1;
            thread_loss[bid] = logf(sum_exp[0]) - logits_value;
//            if (thread_loss[bid] < 0) {
//                printf("test --- thread_loss[%d]:%f\n", bid, thread_loss[bid]);
//            }
        }
        bid += gridDim.x;
        __syncthreads();
    }
}

__global__
void cuda_CrossEntropy_forward_B_kernel(float *logits_grad, int size, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) logits_grad[i] /= count;
}


// ReLU
__global__
void cuda_ReLU_forward_kernel(float *d_in_data, bool *d_mask, const long unsigned int datasize, bool training) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= datasize) return;

    bool keep = d_in_data[i] > 0;
    if (training) d_mask[i] = keep;
    if (!keep) d_in_data[i] = 0;
}

__global__
void cuda_ReLU_backward_kernel(float *d_in_grad, bool *d_mask, long unsigned int datasize) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= datasize) return;
    if (!d_mask[i]) d_in_grad[i] = 0;
}


// Dropout
__global__
void cuda_Dropout_forward_kernel(float *in, int *mask, curandState *state, const uint size, const float p, const float scale, const bool useMask) {
    float x;
    bool keep;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        x = curand_uniform(&state[id % MAX_THREAD_PER_BLOCK]);
        keep = x >= p;
        in[id] *= keep ? scale : 0;
        if (useMask) mask[id] = keep;
    }
}

__global__
void cuda_Dropout_backward_kernel(float *in_grad, const int *mask, const uint size, const float scale) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) in_grad[id] *= mask[id] ? scale : 0;
}


// rand state
__global__
void cuda_init_rand_kernel(curandState *state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, id, 0, &state[id]);
}

void cuda_init_random_state(const uint size) {
    // malloc
    CUDA_CHECK(cudaMalloc((void**) &devStates, size * sizeof(curandState)));

    dim3 block((size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);

    // kernel
    cuda_init_rand_kernel<<<block,thread_in_block>>>(devStates);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_free_random_state() {
    // free
    CUDA_CHECK(cudaFree(devStates));
}


// adam
__global__
void cuda_Adam_step_kernel(float* grad, float* data, float* m, float* v, bool decay, float weight_decay, float beta1, float beta2, float eps, float step_size, int varsize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= varsize) return;

    float g = grad[i];
    if (decay) g += weight_decay * data[i];
    m[i] = beta1 * m[i] + (1.0 - beta1) * g;
    v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
    data[i] -= step_size * m[i] / (sqrtf(v[i]) + eps);
}

__global__
void cuda_set_truth_kernel(int *truth, int *data_split, int *data_label, int current_split, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
        truth[id] = data_split[id] == current_split ? data_label[id] : -1;
}

__global__
void cuda_Variable_glorot_kernel(float *data, curandState *state, int size, float scale) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
        data[id] = (curand_uniform(&state[id % MAX_THREAD_PER_BLOCK]) - 0.5) * scale;
//		data[id] = scale * threadIdx.x / blockDim.x;
}
