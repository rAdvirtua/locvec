#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int start_offset;
    int count;
} ClusterOffset;

int build_ivf_structure(int k_clusters, int dims) {
    FILE* f_data = fopen("offline_embeddings.bin", "rb");
    if (!f_data) return 1;

    fseek(f_data, 0, SEEK_END);
    long file_size = ftell(f_data);
    rewind(f_data);

    int n_vectors = file_size / (dims * sizeof(float));

    float* raw_data = (float*)malloc((size_t)n_vectors * dims * sizeof(float));
    int* labels = (int*)malloc((size_t)n_vectors * sizeof(int));
    float* organized_data = (float*)malloc((size_t)n_vectors * dims * sizeof(float));
    int* write_pointers = (int*)malloc((size_t)k_clusters * sizeof(int));
    ClusterOffset* offsets = (ClusterOffset*)calloc(k_clusters, sizeof(ClusterOffset));

    if (!raw_data || !labels || !organized_data || !write_pointers || !offsets) return 1;

    FILE* f_labels = fopen("ivf_labels.bin", "rb");
    fread(raw_data, sizeof(float), (size_t)n_vectors * dims, f_data);
    fread(labels, sizeof(int), n_vectors, f_labels);
    fclose(f_data);
    fclose(f_labels);

    for (int i = 0; i < n_vectors; ++i) {
        if (labels[i] >= 0 && labels[i] < k_clusters) {
            offsets[labels[i]].count++;
        }
    }

    int current_offset = 0;
    for (int k = 0; k < k_clusters; ++k) {
        offsets[k].start_offset = current_offset;
        write_pointers[k] = current_offset;
        current_offset += offsets[k].count;
    }

    for (int i = 0; i < n_vectors; ++i) {
        int cluster = labels[i];
        int write_idx = write_pointers[cluster];
        for (int d = 0; d < dims; ++d) {
            organized_data[write_idx * dims + d] = raw_data[i * dims + d];
        }
        write_pointers[cluster]++;
    }

    FILE* f_out_data = fopen("ivf_database.bin", "wb");
    FILE* f_out_offsets = fopen("ivf_offsets.bin", "wb");
    
    fwrite(organized_data, sizeof(float), (size_t)n_vectors * dims, f_out_data);
    fwrite(offsets, sizeof(ClusterOffset), k_clusters, f_out_offsets);
    
    fclose(f_out_data);
    fclose(f_out_offsets);

    free(raw_data); free(labels); free(organized_data); free(write_pointers); free(offsets);
    return 0;
}