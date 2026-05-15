#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void optimized_assign_kernel(const float* __restrict__ data_soa, const float* __restrict__ centroids, int* __restrict__ labels, int n_vectors, int k_clusters, int dims) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= n_vectors) return;

    float min_dist = 1e30f;
    int best_cluster = 0;

    for (int k = 0; k < k_clusters; ++k) {
        float dist = 0.0f;
        for (int d = 0; d < dims; ++d) {
            float val = data_soa[d * n_vectors + vec_idx];
            float diff = val - centroids[k * dims + d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = k;
        }
    }
    labels[vec_idx] = best_cluster;
}

__global__ void sum_centroids_kernel_soa(const float* __restrict__ data_soa, const int* __restrict__ labels, float* __restrict__ new_centroids, int* __restrict__ counts, int n_vectors, int dims) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx < n_vectors) {
        int cluster_id = labels[vec_idx];
        atomicAdd(&counts[cluster_id], 1);
        for (int d = 0; d < dims; ++d) {
            atomicAdd(&new_centroids[cluster_id * dims + d], data_soa[d * n_vectors + vec_idx]);
        }
    }
}

__global__ void average_and_check_kernel(float* __restrict__ new_centroids, float* __restrict__ old_centroids, const int* __restrict__ counts, int* __restrict__ d_converged, int k_clusters, int dims) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < k_clusters) {
        int count = counts[k];
        float max_shift = 0.0f;
        if (count > 0) {
            for (int d = 0; d < dims; ++d) {
                int idx = k * dims + d;
                float avg = new_centroids[idx] / count;
                float diff = avg - old_centroids[idx];
                max_shift += diff * diff;
                new_centroids[idx] = avg;
            }
        }
        if (max_shift > 0.0001f) {
            atomicExch(d_converged, 0);
        }
    }
}

extern "C" int train_index_kernel(int k_clusters, int dims) {
    FILE* f = fopen("offline_embeddings.bin", "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    rewind(f);

    int n_vectors = file_size / (dims * sizeof(float));
    size_t data_bytes = (size_t)n_vectors * dims * sizeof(float);
    size_t centroid_bytes = (size_t)k_clusters * dims * sizeof(float);
    size_t label_bytes = (size_t)n_vectors * sizeof(int);
    size_t count_bytes = (size_t)k_clusters * sizeof(int);

    float* h_data_aos = (float*)malloc(data_bytes);
    float* h_data_soa = (float*)malloc(data_bytes);
    float* h_centroids = (float*)malloc(centroid_bytes);

    fread(h_data_aos, sizeof(float), (size_t)n_vectors * dims, f);
    fclose(f);

    for (int i = 0; i < n_vectors; ++i) {
        for (int d = 0; d < dims; ++d) {
            h_data_soa[d * n_vectors + i] = h_data_aos[i * dims + d];
        }
    }
    memcpy(h_centroids, h_data_aos, centroid_bytes);

    float *d_data_soa, *d_centroids, *d_new_centroids;
    int *d_labels, *d_counts, *d_converged;
    
    cudaMalloc(&d_data_soa, data_bytes);
    cudaMalloc(&d_centroids, centroid_bytes);
    cudaMalloc(&d_new_centroids, centroid_bytes);
    cudaMalloc(&d_labels, label_bytes);
    cudaMalloc(&d_counts, count_bytes);
    cudaMalloc(&d_converged, sizeof(int));

    cudaMemcpy(d_data_soa, h_data_soa, data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, centroid_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks_N = (n_vectors + threads - 1) / threads;
    int blocks_K = (k_clusters + threads - 1) / threads;

    int iter = 0;
    while (iter < 100) {
        int h_converged = 1;
        cudaMemcpy(d_converged, &h_converged, sizeof(int), cudaMemcpyHostToDevice);

        optimized_assign_kernel<<<blocks_N, threads>>>(d_data_soa, d_centroids, d_labels, n_vectors, k_clusters, dims);
        cudaMemset(d_new_centroids, 0, centroid_bytes);
        cudaMemset(d_counts, 0, count_bytes);
        
        sum_centroids_kernel_soa<<<blocks_N, threads>>>(d_data_soa, d_labels, d_new_centroids, d_counts, n_vectors, dims);
        average_and_check_kernel<<<blocks_K, threads>>>(d_new_centroids, d_centroids, d_counts, d_converged, k_clusters, dims);
        
        cudaMemcpy(d_centroids, d_new_centroids, centroid_bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(&h_converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);

        iter++;
        if (h_converged) break;
    }

    cudaMemcpy(h_centroids, d_centroids, centroid_bytes, cudaMemcpyDeviceToHost);
    FILE* out_c = fopen("ivf_centroids.bin", "wb");
    fwrite(h_centroids, sizeof(float), (size_t)k_clusters * dims, out_c);
    fclose(out_c);

    int* h_labels = (int*)malloc(label_bytes);
    cudaMemcpy(h_labels, d_labels, label_bytes, cudaMemcpyDeviceToHost);
    FILE* out_l = fopen("ivf_labels.bin", "wb");
    fwrite(h_labels, sizeof(int), n_vectors, out_l);
    fclose(out_l);

    cudaFree(d_data_soa); cudaFree(d_centroids); cudaFree(d_new_centroids);
    cudaFree(d_labels); cudaFree(d_counts); cudaFree(d_converged);
    free(h_data_aos); free(h_data_soa); free(h_centroids); free(h_labels);

    return 0;
}