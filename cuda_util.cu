#include <stdio.h>
#include <stdlib.h>
extern "C" {
#include "cuda_util.h"
#include "fit_tensor_util.h"
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
__global__ void cutoff_log_kernel(float* device_array, float min_signal){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (device_array[thread_id] < min_signal){
        device_array[thread_id] = logf(min_signal);
    }
    else{
        device_array[thread_id] = logf(device_array[thread_id]);
    }
}

extern "C"
//Function to launch cutoff log kernel
float* cutoff_log_cuda(float* input, float min_signal, int array_length){
    padded_float_array* padded_array = pad_array(input, array_length, WARP_SIZE);
    float* device_array = cuda_float_copy_to_gpu(padded_array->values, padded_array->current_length);
    int blocks_in_grid = padded_array->current_length / WARP_SIZE;
    cutoff_log_kernel<<<blocks_in_grid, WARP_SIZE>>>(device_array, min_signal);
    padded_array->values = cuda_float_return_from_gpu(device_array, padded_array->current_length);
    float* result_array = get_array_from_padded_array(padded_array);
    free_cuda_memory(device_array);
    free_padded_array(padded_array);
    return result_array;
}

//kernel to take entire array and exp it
__global__ void exp_kernel(float* cuda_array){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    puts("Index:%d Value:%s", thread_id, cuda_array[thread_id]);
    cuda_array[thread_id] = expf(cuda_array[thread_id]);
    puts("Index:%d Value:%s", thread_id, cuda_array[thread_id]);
}

extern "C"
//Kernel catapult
void exp_cuda(float* input, int array_length){
    padded_float_array* padded_array = pad_array(input, array_length, WARP_SIZE);
    float* device_array = cuda_float_copy_to_gpu(padded_array->values, padded_array->current_length);
    int blocks_in_grid = padded_array->current_length/ WARP_SIZE;
    exp_kernel<<<blocks_in_grid, WARP_SIZE>>>(device_array);
    padded_array->values = cuda_float_return_from_gpu(device_array, padded_array->current_length);
    input = get_array_from_padded_array(padded_array);
    free_cuda_memory(device_array);
    free_padded_array(padded_array);
}

