#include <stdio.h>
#include <stdlib.h>
extern "C" {
#include "cuda_util.h"
}

extern "C"
//clones float array and copies to gpu
float* cuda_float_copy_to_gpu(float* local_array, int array_length){
    float* cuda_array;
    cudaMalloc(&cuda_array, sizeof(float) * array_length);
    cudaMemcpy(cuda_array, local_array, sizeof(float) * array_length, cudaMemcpyHostToDevice);
    return cuda_array;
}

extern "C"
//clones float array and copies to host 
float* cuda_float_return_from_gpu(float* cuda_array, int array_length){
    float* result_array = (float *) malloc(sizeof(float) * array_length);
    cudaMemcpy(result_array, cuda_array, sizeof(float) * array_length, cudaMemcpyDeviceToHost);
    return result_array;
}

extern "C"
//allocates space for a float array on the device
void cuda_float_allocate(float* pointer, int pointer_length){
    cudaMalloc(&pointer, pointer_length);
}

extern "C"
//frees float device memory
void free_cuda_memory(float* pointer){
    cudaFree(pointer);
}

//kernel to take entire array and run cutoff log function
__global__ void cutoff_log_kernel(float* input, float* output, float min_signal){
    int thread_id = blockIdx.x;
    if (input[thread_id] < min_signal){
        output[thread_id] = logf(min_signal);
    }
    else{
        output[thread_id] = logf(input[thread_id]);
    }
}

extern "C"
//Function to launch cutoff log kernel
void cutoff_log_cuda(float* input, float* output, float min_signal, int block_grid_rows){
  cutoff_log_kernel<<<block_grid_rows, 1>>>(input, output, min_signal);
}

//kernel to take entire array and exp it
__global__ void exp_kernel(float* cuda_array){
    int thread_id = blockIdx.x;
    cuda_array[thread_id] = expf(cuda_array[thread_id]);
}

extern "C"
//Kernel catapult
void exp_cuda(float* cuda_array, int block_grid_rows){
  exp_kernel<<<block_grid_rows, 1>>>(cuda_array);
}

