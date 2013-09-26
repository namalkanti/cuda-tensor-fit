#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32

//clones float array and copies to gpu
float* cuda_float_copy_to_gpu(float* local_array, int array_length);

//clones float array and copies to host 
float* cuda_float_return_from_gpu(float* cuda_array, int array_length);

//allocates space for a float array on the device
void cuda_float_allocate(float* pointer, int pointer_length);

//frees float device memory
void free_cuda_memory(float* pointer);

//kernel to take entire array and run cutoff log function
void cutoff_log_cuda(float* input, float* output, float min_signal, int block_grid_rows);

//kernel to take entire array and exp it
void exp_cuda(float* cuda_array, int block_grid_rows);
