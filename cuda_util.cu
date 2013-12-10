#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
extern "C" {
#include "cuda_util.h"
}
#include "BatchedSolver/solve.h"
#define IDX2C(i, j, ld) ((j)*(ld)+(i))

//Helper function declarations
double* convert_matrix_to_fortran_and_load_to_gpu(matrix* mat);
void get_matrix_from_gpu_and_convert_from_fortran(double* gpu_pointer, matrix* mat);

//Kernel declarations
__global__ void cutoff_log_kernel(double* device_array, double min_signal);
__global__ void exp_kernel(double* cuda_array);
__global__ void weighting_kernel (double* matrices, double* weights); 
__global__ void weighting_kernel_transposed(double* matrices, double* weights); 



extern "C"
double* cuda_double_copy_to_gpu(double* local_array, int array_length){
    double* cuda_array;
    cudaMalloc(&cuda_array, sizeof(double) * array_length);
    cudaMemcpy(cuda_array, local_array, sizeof(double) * array_length, cudaMemcpyHostToDevice);
    return cuda_array;
}

extern "C"
double* cuda_double_return_from_gpu(double* cuda_array, int array_length){
    double* result_array = (double *) malloc(sizeof(double) * array_length);
    cudaMemcpy(result_array, cuda_array, sizeof(double) * array_length, cudaMemcpyDeviceToHost);
    return result_array;
}

extern "C"
void cuda_double_allocate(double* pointer, int pointer_length){
    cudaMalloc(&pointer, pointer_length);
}

extern "C"
void free_cuda_memory(double* pointer){
    cudaFree(pointer);
}

extern "C"
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

extern "C"
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
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        puts("Failed to retrieve cublas handle.");
    }
    double* gpu_array1 = convert_matrix_to_fortran_and_load_to_gpu(matrix1);
    double* gpu_array2 = convert_matrix_to_fortran_and_load_to_gpu(matrix2);
    double* gpu_output;
    cudaMalloc(&gpu_output, sizeof(double)* matrix1->rows * matrix2->columns);
    const double alpha = 1;
    const double beta = 0;
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix1->rows, matrix2->columns, matrix1->columns, 
            &alpha, gpu_array1, matrix1->rows, gpu_array2, matrix2->rows, &beta, gpu_output, matrix1->rows);
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        puts("Call to cublas function failed.");
    }
    matrix* result_matrix = (matrix*) malloc(sizeof(matrix));
    double* result_matrix_data = (double*) malloc(sizeof(double) * matrix1->rows * matrix2->columns);
    result_matrix->rows = matrix1->rows;
    result_matrix->columns = matrix2->columns;
    result_matrix->data = result_matrix_data;
    get_matrix_from_gpu_and_convert_from_fortran(gpu_output, result_matrix);
    cudaFree(gpu_array1);
    cudaFree(gpu_array2);
    return result_matrix;
}
    
extern "C"
void matrix_weighter (double* matrices, double* weights, int rows, int columns, int length, bool trans) {
    dim3 grid, block;
    int weight_length;
    grid.x = length;
    block.x = columns;
    block.y = rows;
    if ( false == trans ) {
        weight_length = columns;
    }
    else {
        weight_length = rows;
    }
    double* gpu_matrices = cuda_double_copy_to_gpu(matrices, rows * columns * length);
    double* gpu_weights = cuda_double_copy_to_gpu(weights, weight_length);
    if (false == trans){
        weighting_kernel<<<grid, block>>>(gpu_matrices, gpu_weights);
    }
    else {
        weighting_kernel_transposed<<<grid, block>>>(gpu_matrices, gpu_weights);
    }
    cudaMemcpy(matrices, gpu_matrices, sizeof(double) * rows* columns * length, cudaMemcpyDeviceToHost);
    cudaFree(gpu_matrices);
    cudaFree(gpu_weights);
}

extern "C"
double* cuda_fitter(matrix const* design_matrix, matrix const* weights, double const* signal, int signal_length){
}

extern "C"
void decompose_tensors(double const* tensors, tensor** tensor_output){
}

//Helper functions

/*Converts matrix to the data format fortran uses for CUBLAS and loads to GPU
  Returns pointer to array on GPU.*/
double* convert_matrix_to_fortran_and_load_to_gpu(matrix* mat){
    cublasStatus_t status;
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
    status = cublasSetMatrix(mat->rows, mat->columns, sizeof(double), intermediate_matrix, 
            mat->rows, gpu_pointer, mat->rows);
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        puts("Failed to copy matrix to memory.");
    }
    free(intermediate_matrix);
    return gpu_pointer;
}

/*Converts matrix from the format fortran uses for CUBLAS after retrieving from GPU
  Will free gpu_pointer.
  Populates a matrix object passed in.*/
void get_matrix_from_gpu_and_convert_from_fortran(double* gpu_pointer, matrix* mat){
    cublasStatus_t status;
    int length = mat->rows * mat->columns;
    double* intermediate_matrix = (double*) malloc(sizeof(double) * length);
    status = cublasGetMatrix(mat->rows, mat->columns, sizeof(double), gpu_pointer, mat->rows,
            intermediate_matrix, mat->rows);
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        puts("Failed to retrieve matrix from memory.");
    }
    int i, j;
    for (i = 0; i < mat->rows; i++ ) {
        for (j = 0; j < mat->columns; j++) {
            mat->data[i * mat->rows + j] = intermediate_matrix[IDX2C(i, j, mat->rows)];
        }
    }
    free(intermediate_matrix);
}

//Kernels

//kernel to take entire array and run cutoff log function
__global__ void cutoff_log_kernel(double* device_array, double min_signal){
    int thread_id = blockidx.x * blockdim.x + threadidx.x;
    if (device_array[thread_id] < min_signal){
        device_array[thread_id] = logf(min_signal);
    }
    else{
        device_array[thread_id] = logf(device_array[thread_id]);
    }
}

//kernel to take entire array and exp it
__global__ void exp_kernel(double* cuda_array){
    int thread_id = blockidx.x * blockdim.x + threadidx.x;
    cuda_array[thread_id] = expf(cuda_array[thread_id]);
}

//kernel for weighting the matrix.
__global__ void weighting_kernel (double* matrices, double* weights) {
    int grid_index = blockidx.x * blockdim.x * blockdim.y;
    int block_index = blockdim.y * threadidx.y + threadidx.x;
    int matrix_index = grid_index + block_index;
    matrices[matrix_index] = matrices[matrix_index] * weights[threadidx.x];
}

//kernel for weighting a transposed matrix.
__global__ void weighting_kernel_transposed(double* matrices, double* weights) {
    int grid_index = blockidx.x * blockdim.x * blockdim.y;
    int block_index = blockdim.y * threadidx.y + threadidx.x;
    int matrix_index = grid_index + block_index;
    int weighting_index = blockidx.x * blockdim.y + threadidx.y; 
    matrices[matrix_index] = matrices[matrix_index] * weights[weighting_index];
}

