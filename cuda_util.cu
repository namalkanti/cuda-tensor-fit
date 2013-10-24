#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
extern "C" {
#include "cuda_util.h"
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

/*Converts matrix to the data format fortran uses for CUBLAS and loads to GPU
  Returns pointer to array on GPU.*/
double* convert_matrix_to_fortran_and_load_to_gpu(matrix* mat){
    int length = mat->rows * mat->columns;
    double* gpu_pointer; 
    double* intermediate_matrix = (double*) malloc(sizeof(double) * length);
    cudaMalloc(&gpu_pointer, sizeof(double) * length);
    int i, j;
    for (i = 0; i < mat->rows; i++ ) {
        for (j = 0; j < mat->columns; j++) {
            intermediate_matrix[IDX2C(i, j, mat->rows)] = mat->data[i * mat->rows + j];
        }
    }
    cublasSetMatrix(mat->rows, mat->columns, sizeof(double), intermediate_matrix, 
            mat->rows, gpu_pointer, mat->rows);
    free(intermediate_matrix);
    return gpu_pointer;
}

/*Converts matrix from the format fortran uses for CUBLAS after retrieving from GPU
  Will free gpu_pointer.
  Populates a matrix object passed in.*/
void get_matrix_from_gpu_and_convert_from_fortran(double* gpu_pointer, matrix* mat){
    int length = mat->rows * mat->columns;
    double* intermediate_matrix = (double*) malloc(sizeof(double) * length);
    cudaGetMatrix(mat->rows, mat->columns, sizeof(double), gpu_pointer, mat->rows,
            intermediate_matrix, mat->rows);
    int i, j;
    for (i = 0; i < mat->rows; i++ ) {
        for (j = 0; j < mat->columns; j++) {
            mat->data[i * mat->rows + j] = intermediate_matrix[IDX2C(i, j, mat->rows)];
        }
    }
    cudaFree(gpu_pointer);
    free(intermediate_matrix);
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
    double* gpu_array1 = convert_matrix_to_fortran_and_load_to_gpu(matrix1);
    double* gpu_array2 = convert_matrix_to_fortran_and_load_to_gpu(matrix2);
    double* gpu_output = (double*) calloc(matrix1->rows * matrix2->columns, sizeof(double));
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix1->rows, matrix2->columns, matrix1->columns, 
            const 1.0, gpu_array1, matrix1->rows, gpu_array2, matrix2->rows, const 0.0, gpu_output, matrix1->rows);
    matrix* result_matrix = (matrix*) malloc(sizeof(matrix));
    result_matrix->rows = matrix1->rows;
    result_matrix->columns = matrix2->columns;
    get_matrix_from_gpu_and_convert_from_fortran(gpu_output, result_matrix);
    return result_matrix;
}
