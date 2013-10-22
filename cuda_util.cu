#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
extern "C" {
#include "cuda_util.h"
#include "fit_tensor_util.h"
}

extern "C"
//clones double array and copies to gpu
double* cuda_double_copy_to_gpu(double* local_array, int array_length){
    double* cuda_array;
    cudaMalloc(&cuda_array, sizeof(double) * array_length);
    cudaMemcpy(cuda_array, local_array, sizeof(double) * array_length, cudaMemcpyHostToDevice);
    return cuda_array;
}

extern "C"
//clones double array and copies to host 
double* cuda_double_return_from_gpu(double* cuda_array, int array_length){
    double* result_array = (double *) malloc(sizeof(double) * array_length);
    cudaMemcpy(result_array, cuda_array, sizeof(double) * array_length, cudaMemcpyDeviceToHost);
    return result_array;
}

extern "C"
//allocates space for a double array on the device
void cuda_double_allocate(double* pointer, int pointer_length){
    cudaMalloc(&pointer, pointer_length);
}

extern "C"
//frees double device memory
void free_cuda_memory(double* pointer){
    cudaFree(pointer);
}

//kernel to take entire array and run cutoff log function
__global__ void cutoff_log_kernel(double* device_array, double min_signal){
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
double* cutoff_log_cuda(double* input, double min_signal, int array_length){
    padded_double_array* padded_array = pad_array(input, array_length, WARP_SIZE);
    double* device_array = cuda_double_copy_to_gpu(padded_array->values, padded_array->current_length);
    int blocks_in_grid = padded_array->current_length / WARP_SIZE;
    cutoff_log_kernel<<<blocks_in_grid, WARP_SIZE>>>(device_array, min_signal);
    padded_array->values = cuda_double_return_from_gpu(device_array, padded_array->current_length);
    double* result_array = get_array_from_padded_array(padded_array);
    free_cuda_memory(device_array);
    free_padded_array(padded_array);
    return result_array;
}

//kernel to take entire array and exp it
__global__ void exp_kernel(double* cuda_array){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    cuda_array[thread_id] = expf(cuda_array[thread_id]);
}

extern "C"
//Kernel catapult
double* exp_cuda(double* input, int array_length){
    padded_double_array* padded_array = pad_array(input, array_length, WARP_SIZE);
    double* device_array = cuda_double_copy_to_gpu(padded_array->values, padded_array->current_length);
    int blocks_in_grid = padded_array->current_length/ WARP_SIZE;
    exp_kernel<<<blocks_in_grid, WARP_SIZE>>>(device_array);
    padded_array->values = cuda_double_return_from_gpu(device_array, padded_array->current_length);
    double* output_array = get_array_from_padded_array(padded_array);
    free_cuda_memory(device_array);
    free_padded_array(padded_array);
    return output_array;
}


extern "C"
matrix* cuda_matrix_dot(matrix* matrix1, matrix* matrix2){
}
