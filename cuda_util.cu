#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
extern "C" {
#include "cuda_util.h"
}
#include "BatchedSolver/solve.h"
#define IDX2C(i, j, ld) ((j)*(ld)+(i))

//Helper function declarations
double* convert_matrix_to_fortran_and_load_to_gpu(matrix const* mat);
void get_matrix_from_gpu_and_convert_from_fortran(double const* gpu_pointer, matrix* mat);

//Kernel declarations
__global__ void cutoff_log_kernel(double* device_array, double min_signal);
__global__ void exp_kernel(double* cuda_array);
__global__ void weighting_kernel (double* matrices, double* weights, double* results); 
__global__ void weighting_kernel_transposed(double* matrices, double* weights, double* results); 
__global__ void transpose_kernel(double const* matrices, double* transposed);


extern "C"
matrix* process_signal(matrix const* signal){
}

extern "C"
double* cuda_double_copy_to_gpu(double const* local_array, int array_length){
    double* cuda_array;
    cudaMalloc(&cuda_array, sizeof(double) * array_length);
    cudaMemcpy(cuda_array, local_array, sizeof(double) * array_length, cudaMemcpyHostToDevice);
    return cuda_array;
}

extern "C"
double* cuda_double_return_from_gpu(double const* cuda_array, int array_length){
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
void free_matrix_with_cuda_pointer(matrix* gpu_matrix){
    free_cuda_memory(gpu_matrix->data);
    free(gpu_matrix);
}

extern "C"
double* cutoff_log_cuda(double const* input, double min_signal, int array_length){
    padded_array* padded_arr = pad_array(input, array_length, WARP_SIZE);
    double* device_array = cuda_double_copy_to_gpu(padded_arr->values, padded_arr->current_length);
    int blocks_in_grid = padded_arr->current_length / WARP_SIZE;
    cutoff_log_kernel<<<blocks_in_grid, WARP_SIZE>>>(device_array, min_signal);
    padded_arr->values = cuda_double_return_from_gpu(device_array, padded_arr->current_length);
    double* result_array = get_array_from_padded_array(padded_arr);
    free_cuda_memory(device_array);
    free_padded_array(padded_arr);
    return result_array;
}

extern "C"
double* exp_cuda(double const* input, int array_length){
    padded_array* padded_arr = pad_array(input, array_length, WARP_SIZE);
    double* device_array = cuda_double_copy_to_gpu(padded_arr->values, padded_arr->current_length);
    int blocks_in_grid = padded_arr->current_length/ WARP_SIZE;
    exp_kernel<<<blocks_in_grid, WARP_SIZE>>>(device_array);
    padded_arr->values = cuda_double_return_from_gpu(device_array, padded_arr->current_length);
    double* output_array = get_array_from_padded_array(padded_arr);
    free_cuda_memory(device_array);
    free_padded_array(padded_arr);
    return output_array;
}

extern "C"
matrix* cuda_matrix_dot(matrix const* matrix1, matrix const* matrix2){
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
double* matrix_weighter (double const* matrix, double const* weights, int rows, int columns, int length, bool trans) {
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
    double* gpu_matrix = cuda_double_copy_to_gpu(matrix, rows * columns);
    double* gpu_weights = cuda_double_copy_to_gpu(weights, weight_length * length);
    double* gpu_results;
    cudaMalloc(&gpu_results, sizeof(double) * rows * columns * length);
    if (false == trans){
        weighting_kernel<<<grid, block>>>(gpu_matrix, gpu_weights, gpu_results);
    }
    else {
        weighting_kernel_transposed<<<grid, block>>>(gpu_matrix, gpu_weights, gpu_results);
    }
    double* weighted_matrices = malloc(sizeof(double) * rows * columns * length);
    cudaMemcpy(weighted_matrices, gpu_results, sizeof(double) * rows * columns * length, cudaMemcpyDeviceToHost);
    cudaFree(gpu_matrix);
    cudaFree(gpu_weights);
    cudaFree(gpu_results);
    return weighted_matrices;
}

extern "C"
double* transpose_matrices(double* matrices, int rows, int columns, int length){
    double* transposed = malloc(sizeof(double) * rows * columns * length);
    double* gpu_matrices = cuda_double_copy_to_gpu(matrices, rows * columns * length);
    double* gpu_tranposed = cuda_double_copy_to_gpu(transposed, rows * columns * length);
    dim3 grid, block;
    grid.x = length;
    block.x = columns;
    block.y = rows;
    transpose_kernel<<<grid, block>>>(double const* gpu_matrices, double* gpu_transposed);
    transposed = cuda_double_return_from_gpu(gpu_transposed, rows * columns * length);
    free_cuda_memory(gpu_matrices);
    free_cuda_memory(gpu_transposed);
    return transposed;
}

extern "C"
double* dot_matrices(double const* matrix_batch_one, int rows, double const* matrix_batch_two, int columns,
        int k, int length){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        puts("Failed to retrieve cublas handle.");
    }
    double* transposed_batch1 = transpose_matrices(matrix_batch_one, rows, k, length);
    double* transposed_batch2 = transpose_matrices(matrix_batch_two, k, columns, length);
    double* gpu_array1 = cuda_double_copy_to_gpu(transposed_batch1, rows * k * length);
    double* gpu_array2 = cuda_double_copy_to_gpu(transposed_batch2, k *  columns * length);
    double* gpu_output;
    cudaMalloc(&gpu_output, sizeof(double)* transposed_batch1->rows 
            * transposed_batch2->columns * length);
    const double alpha = 1;
    const double beta = 0;
    status = cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, columns, 
            k, &alpha, gpu_array1, rows, gpu_array2, k, &beta, 
            gpu_output, rows, length);
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        puts("Call to cublas function failed.");
    }
    double* results;
    cudaMalloc(&gpu_output, sizeof(double) * rows * columns * length);
    results = cuda_double_return_from_gpu(gpu_output, rows * columns * length);
    results = transpose_matrices(results, rows, columns, length);
    free_cuda_memory(gpu_array1);
    free_cuda_memory(gpu_array2);
    free_cuda_memory(gpu_output);
    free(transposed_batch1);
    free(transposed_batch2);
    return results;

}

extern "C"
double* solve_matrices(){
}

extern "C"
double* cuda_fitter(matrix const* design_matrix, matrix const* column_major_weights, 
        double const* signal, int signal_length, int number_of_signals){
    int signal_elements = signal_length;
    int total_elements = signal_elements * number_of_signals;
    double* cutoff_and_logged_signal = cutoff_log_cuda(signal, min_signal, total_elements);
    matrix signal_matrix = {cutoff_and_logged_signal, signal_elements, signal_length};
    matrix* ols_signal_dot_matrix = cuda_matrix_dot(ols_matrix, &signal_matrix);
    matrix* weights = exp_cuda(ols_signal_dot_matrix, total_elements);
    matrix* weighted_matrices = matrix_weighter(signal, weights, signal_elements, signal_length, total_elements, false);
    matrix* transposed_weighted_matrices = transpose_matrices();
    matrix* column_major_data = cuda_matrix_dot(transposed_weighted_matrices, signal);
    matrix* data = transpose_matrices(column_major_data);
    matrix* weighted_fitting_matrix = dot_matrices;
    matrix* solutions = solve_matrices();
}

extern "C"
void decompose_tensors(double const* tensors, tensor** tensor_output){
}

//Helper functions

/*Converts matrix to the data format fortran uses for CUBLAS and loads to GPU
  Returns pointer to array on GPU.*/
double* convert_matrix_to_fortran_and_load_to_gpu(matrix const* mat){
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
void get_matrix_from_gpu_and_convert_from_fortran(double const* gpu_pointer, matrix* mat){
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
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (device_array[thread_id] < min_signal){
        device_array[thread_id] = logf(min_signal);
    }
    else{
        device_array[thread_id] = logf(device_array[thread_id]);
    }
}

//kernel to take entire array and exp it
__global__ void exp_kernel(double* cuda_array){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    cuda_array[thread_id] = expf(cuda_array[thread_id]);
}

//kernel for weighting the matrix.
__global__ void weighting_kernel (double* matrix, double* weights, double* results) {
    int matrix_grid_index = blockIdx.x * blockDim.x * blockDim.y;
    int block_index = blockDim.y * threadIdx.y + threadIdx.x;
    int matrix_index = grid_index + block_index;
    int weight_index = blockIdx.x * blockDim.x + threadIdx.x; 
    results[matrix_index] = matrices[block_index] * weights[weight_index];
}

//kernel for weighting a transposed matrix.
__global__ void weighting_kernel_transposed(double* matrix, double* weights, double* results) {
    int grid_index = blockIdx.x * blockDim.x * blockDim.y;
    int block_index = blockDim.y * threadIdx.y + threadIdx.x;
    int matrix_index = grid_index + block_index;
    int weighting_index = blockIdx.x * blockDim.y + threadIdx.y; 
    results[matrix_index] = matrices[block_index] * weights[weighting_index];
}

//kernel for transposing multiple matrices.
__global__ void transpose_kernel(double const* matrices, double* transposed) {
    int matrix_offset = blockIdx.x * blockDim.x * blockDim.y;
    int matrix_index = matrix_offset + blockDim.x * threadIdx.y + threadIdx.x;
    int transpose_index = matrix_offset + IDX2C(threadIdx.x, threadIdx.y, blockDim.y);
    transposed[transpose_index] = matrices[matrix_index];
}

