#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
extern "C" {
#include "cuda_util.h"
#include "fit_tensor_util.h"
}
#define IDX2C(i, j, ld) ((j)*(ld)+(i))

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

//Converts matrix to the data format fortran uses for CUBLAS
void convert_matrix_to_fortran(){
}

//Converts matrix from the format fortran uses for CUBLAS
void convert_matrix_from_fortran(){
}

extern "C"
//Returns the dot product of the two matrices, does calculations on the GPU
matrix* cuda_matrix_dot(matrix* matrix1, matrix* matrix2){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    if ( status != CUBLAS_STATUS_SUCESS ) {
        puts("Failed to retrieve cublas handle");
    }
    convert_matrix_to_fortran(matrix1->data);
    convert_matrix_to_fortran(matrix2->data);
    double* output = calloc(sizeof(double) * matrix1->rows * matrix2->columns);
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix1->rows, matrix2->columns, matrix1->columns, 
            1.0, matrix1->data, , matrix2->data, , 0, output, ,);
    matrix* result_matrix = malloc(sizeof(matrix));
    convert_matrix_from_fortran(output);
    result_matrix->data = output;
    result_matrix->rows = matrix1->rows;
    result_matrix->columns = matrix2->columns;
    return result_matrix;
}
