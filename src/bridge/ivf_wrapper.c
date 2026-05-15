#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int start_offset;
    int count;
} ClusterOffset;

float* global_centroids = NULL;
ClusterOffset* global_offsets = NULL;
int global_k = 0;
int total_vectors_indexed = 0;

extern int run_cluster_search(float* cluster_data, float* query, int count, int dims);
extern int train_index_kernel(int k, int dims);
extern int build_ivf_structure(int k, int dims);

#ifdef __cplusplus
extern "C" {
#endif

int init_engine(int dims) {
    FILE* fc = fopen("ivf_centroids.bin", "rb");
    if (!fc) return -1;
    fseek(fc, 0, SEEK_END);
    global_k = ftell(fc) / (dims * sizeof(float));
    rewind(fc);

    global_centroids = (float*)malloc(global_k * dims * sizeof(float));
    fread(global_centroids, sizeof(float), global_k * dims, fc);
    fclose(fc);

    global_offsets = (ClusterOffset*)malloc(global_k * sizeof(ClusterOffset));
    FILE* fo = fopen("ivf_offsets.bin", "rb");
    if (!fo) {
        free(global_centroids);
        global_centroids = NULL;
        return -2;
    }
    fread(global_offsets, sizeof(ClusterOffset), global_k, fo);
    fclose(fo);

    total_vectors_indexed = 0;
    for(int i = 0; i < global_k; i++) {
        total_vectors_indexed += global_offsets[i].count;
    }
    return 0;
}

void cleanup_engine() {
    if (global_centroids) { free(global_centroids); global_centroids = NULL; }
    if (global_offsets) { free(global_offsets); global_offsets = NULL; }
}

int train_index(int k, int dims) {
    return train_index_kernel(k, dims);
}

int build_index(int k, int dims) {
    return build_ivf_structure(k, dims);
}

int vector_search(float* query, int dims) {
    if (!global_centroids || !global_offsets) return -404;

    float min_dist = 1e30f;
    int best_cluster = -1;

    for (int k = 0; k < global_k; ++k) {
        float dist = 0.0f;
        for (int d = 0; d < dims; ++d) {
            float diff = query[d] - global_centroids[k * dims + d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = k;
        }
    }

    if (best_cluster == -1 || global_offsets[best_cluster].count <= 0) return -1;

    int target_offset = global_offsets[best_cluster].start_offset;
    int target_count = global_offsets[best_cluster].count;

    float* cluster_data = (float*)malloc(target_count * dims * sizeof(float));
    FILE* fd = fopen("ivf_database.bin", "rb");
    if (!fd) { free(cluster_data); return -2; }
    
    fseek(fd, (size_t)target_offset * dims * sizeof(float), SEEK_SET);
    fread(cluster_data, sizeof(float), (size_t)target_count * dims, fd);
    fclose(fd);

    int result_idx = run_cluster_search(cluster_data, query, target_count, dims);
    free(cluster_data);

    return target_offset + result_idx;
}

#ifdef __cplusplus
}
#endif