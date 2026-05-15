#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void cluster_search_kernel(const float* __restrict__ cluster_data, const float* __restrict__ query, float* __restrict__ scores, int count, int dims) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx < count) {
        float dist = 0.0f;
        for (int i = 0; i < dims; ++i) {
            float diff = cluster_data[vec_idx * dims + i] - query[i];
            dist += diff * diff;
        }
        scores[vec_idx] = dist;
    }
}

extern "C" int run_cluster_search(float* cluster_data, float* query, int count, int dims) {
    float *d_data, *d_query, *d_scores;
    float *h_scores = (float*)malloc(count * sizeof(float));
    if (!h_scores) return -1;

    if (cudaMalloc(&d_data, (size_t)count * dims * sizeof(float)) != cudaSuccess) {
        free(h_scores);
        return -1;
    }
    if (cudaMalloc(&d_query, (size_t)dims * sizeof(float)) != cudaSuccess) {
        cudaFree(d_data);
        free(h_scores);
        return -1;
    }
    if (cudaMalloc(&d_scores, (size_t)count * sizeof(float)) != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_query);
        free(h_scores);
        return -1;
    }

    cudaMemcpy(d_data, cluster_data, (size_t)count * dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, (size_t)dims * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (count + threads - 1) / threads;

    cluster_search_kernel<<<blocks, threads>>>(d_data, d_query, d_scores, count, dims);
    
    cudaMemcpy(h_scores, d_scores, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost);

    int best_idx = 0;
    float min_score = 1e30f;
    for(int i = 0; i < count; i++) {
        if(h_scores[i] < min_score) {
            min_score = h_scores[i];
            best_idx = i;
        }
    }

    cudaFree(d_data);
    cudaFree(d_query);
    cudaFree(d_scores);
    free(h_scores);

    return best_idx;
}