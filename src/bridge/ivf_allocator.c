#include <stdio.h>
#include <stdlib.h>

#define DIMS 384
#define K_CLUSTERS 1024

typedef struct {
    int start_offset;
    int count;
} ClusterOffset;

int build_ivf_structure() {
    FILE* f_data = fopen("offline_embeddings.bin", "rb");
    if (!f_data) return 1;

    fseek(f_data, 0, SEEK_END);
    long file_size = ftell(f_data);
    rewind(f_data);

    int n_vectors = file_size / (DIMS * sizeof(float));

    size_t data_bytes = (size_t)n_vectors * DIMS * sizeof(float);
    size_t label_bytes = (size_t)n_vectors * sizeof(int);

    float* raw_data = (float*)malloc(data_bytes);
    int* labels = (int*)malloc(label_bytes);
    float* organized_data = (float*)malloc(data_bytes);
    int* write_pointers = (int*)malloc(K_CLUSTERS * sizeof(int));

    if (!raw_data || !labels || !organized_data || !write_pointers) return 1;

    FILE* f_labels = fopen("ivf_labels.bin", "rb");
    if (!f_labels) return 1;
    
    fread(raw_data, sizeof(float), (size_t)n_vectors * DIMS, f_data);
    fread(labels, sizeof(int), n_vectors, f_labels);
    fclose(f_data);
    fclose(f_labels);

    ClusterOffset offsets[K_CLUSTERS] = {0};
    for (int i = 0; i < n_vectors; ++i) {
        if (labels[i] >= 0 && labels[i] < K_CLUSTERS) {
            offsets[labels[i]].count++;
        }
    }

    int current_offset = 0;
    for (int k = 0; k < K_CLUSTERS; ++k) {
        offsets[k].start_offset = current_offset;
        write_pointers[k] = current_offset;
        current_offset += offsets[k].count;
    }

    for (int i = 0; i < n_vectors; ++i) {
        int cluster = labels[i];
        int write_idx = write_pointers[cluster];
        for (int d = 0; d < DIMS; ++d) {
            organized_data[write_idx * DIMS + d] = raw_data[i * DIMS + d];
        }
        write_pointers[cluster]++;
    }

    FILE* f_out_data = fopen("ivf_database.bin", "wb");
    FILE* f_out_offsets = fopen("ivf_offsets.bin", "wb");
    
    if (f_out_data && f_out_offsets) {
        fwrite(organized_data, sizeof(float), (size_t)n_vectors * DIMS, f_out_data);
        fwrite(offsets, sizeof(ClusterOffset), K_CLUSTERS, f_out_offsets);
        fclose(f_out_data);
        fclose(f_out_offsets);
    }

    free(raw_data); 
    free(labels); 
    free(organized_data); 
    free(write_pointers);

    return 0;
}
