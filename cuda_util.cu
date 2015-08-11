#include <stdlib.h>
#include <cublas_v2.h>
extern "C" {
#include "cuda_util.h"
}
#include "BatchedSolver/solve.h"

#define IDX2C(i, j, ld) ((j)*(ld)+(i))
#define SQR(x)      ((x)*(x))                        // x^2 

#define gpu_error_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, char* file, int line, bool abort=false){
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
    if (abort){
        exit(code);
    }
}

#define WARP_SIZE 16

#define TENSOR_DIMENSIONS 3
#define TENSOR_INPUT_ELEMENTS 6
#define TENSOR_ELEMENTS 9
#define EIGENDECOMPOSITION_ELEMENTS 12


//Helper function declarations
void get_matrix_from_gpu_and_convert_from_fortran(double const* gpu_pointer, matrix* mat);
double** convert_contigous_gpu_array_to_gpu_array_of_pointers(double* arr, int m, int n, int batch, double** intermediate_array);
void free_array_of_gpu_pointers(double** array, int batch);
const char* cublas_get_error_string(cublasStatus_t status);

//Kernel declarations
__global__ void cutoff_log_kernel(double* device_array, double min_signal);
__global__ void exp_kernel(double* cuda_array);
__global__ void weighting_kernel (double const* matrices, double const* weights, double* results); 
__global__ void weighting_kernel_transposed(double const* matrices, double const* weights, double* results); 
__global__ void transpose_kernel(double const* matrices, double* transposed);
__global__ void assemble_tensors(double const* tensor_input, double* tensors, int tensor_input_elements);
__global__ void eigendecomposition_kernel(double const* data, double* eigendecomposition);
__global__ void multiply_arrays(double* signals, double const* weights);
__global__ void create_array_of_pointers_kernel(double* data, int m, int n, double** target);

//device functions
__device__ void assemble_eigendecomposition(double* eigendecomposition, int offset, double Q[3][3], double w[3]);
__device__ void deposit_data_segment_into_array(double const* data, int offset, double A[3][3]);
__device__ int dsyevj3(double A[3][3], double Q[3][3], double w[3]);

extern "C"
matrix* process_signal(matrix const* signal, double min_signal){
    double* signal_data = array_clone(signal->data, signal->rows * signal->columns);
    int total_elements = signal->rows * signal->columns;
    double* gpu_signal = cuda_double_copy_to_gpu(signal_data, total_elements);
    double* kernel_results = cutoff_log_cuda(gpu_signal, min_signal, signal->rows, signal->columns);
    matrix* processed_signal = create_matrix(kernel_results, signal->rows, signal->columns);
    free(signal_data);
    return processed_signal;
}

extern "C"
matrix* generate_weights(matrix const* ols_fit_matrix, matrix const* signal){
    double* gpu_ols_data = convert_matrix_to_fortran_and_load_to_gpu(ols_fit_matrix);
    matrix gpu_ols = {gpu_ols_data, ols_fit_matrix->rows, ols_fit_matrix->columns};
    matrix* weights = cuda_matrix_dot(&gpu_ols, signal);
    double* exp_weights = exp_cuda(weights->data, weights->rows,  weights->columns);
    matrix* gpu_weights= create_matrix(exp_weights, weights->rows, weights->columns);
    return gpu_weights;
}

extern "C"
double* cuda_test_batched_ls(double* ls_matrix, int rows1, int cols1, double* solutions, int rows2, int cols2, int batch_size){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate_v2(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        puts(cublas_get_error_string(status));
    }

    int A_elements = rows1 * cols1;
    int C_elements = rows2 * cols2;

    int* info = (int*) malloc(sizeof(int));
    int* dev_info;
    cudaMalloc(&dev_info, sizeof(int) * batch_size);

    double* ls_gpu = cuda_double_copy_to_gpu(ls_matrix, A_elements * batch_size);
    double* sol_gpu = cuda_double_copy_to_gpu(solutions, C_elements * batch_size);

    double** A_inter;
    A_inter = (double**) malloc(sizeof(double*) * batch_size);
    double** C_inter;
    C_inter = (double**) malloc(sizeof(double*) * batch_size);

    double** A_gpu = convert_contigous_gpu_array_to_gpu_array_of_pointers(ls_gpu, rows1, cols1, batch_size, A_inter);
    double** C_gpu = convert_contigous_gpu_array_to_gpu_array_of_pointers(sol_gpu, rows2, cols2, batch_size, C_inter);

    status = cublasDgelsBatched(handle, CUBLAS_OP_N, rows1, cols1, cols2, A_gpu, rows1,
            C_gpu, rows2, info, dev_info, batch_size);
    if (status != CUBLAS_STATUS_SUCCESS) {
        puts(cublas_get_error_string(status));
    }

    double** sol_array;
    sol_array = (double**) malloc(sizeof(double*) * batch_size);
    gpu_error_check(cudaMemcpy(sol_array, C_gpu, sizeof(double*) * batch_size, cudaMemcpyDeviceToHost));
    double* result;
    double* results = (double*) malloc(sizeof(double) * cols1 * batch_size);
    int i, j, C_offset;
    for(i = 0;i < batch_size;i++){
        C_offset = i * cols1 * cols2 ;
        result = cuda_double_return_from_gpu(sol_array[i], C_elements);
        for(j = 0;j < cols1;j++){
            results[C_offset + j] = result[j];
        }
        free(result);
    }

    status = cublasDestroy_v2(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        puts(cublas_get_error_string(status));
    }

    free_cuda_memory(ls_gpu);
    free_cuda_memory(sol_gpu);
    gpu_error_check(cudaFree(A_gpu));
    gpu_error_check(cudaFree(C_gpu));
    gpu_error_check(cudaFree(dev_info));
    free_array_of_gpu_pointers(A_inter, batch_size);
    free_array_of_gpu_pointers(C_inter, batch_size);
    free(info);
    free(sol_array);
    return results;
}

extern "C"
double* cuda_fitter(matrix const* design_matrix, matrix const* column_major_weights, matrix const* signals){
    //Will not transpose matrix weighting because design matrix is column major already
    double* weighted_design_data = matrix_weighter(design_matrix->data, column_major_weights->data, 
            design_matrix->rows, design_matrix->columns, column_major_weights->columns, false);

    int signal_elements = signals->rows * signals->columns;
    int batch_size = signals->rows;
    int signal_size = signals->columns;

    multiply_arrays<<<batch_size, signal_size>>>(signals->data, column_major_weights->data);
    double* solution_vectors = signals->data;

    /* double* intermediate_solution = cuda_double_return_from_gpu(signals->data, signal_elements); */
    /* double* weights = cuda_double_return_from_gpu(column_major_weights->data, signal_elements); */
    /* int i; */
    /* for(i = 0; i < signal_elements;i++){ */
    /*     intermediate_solution[i] *= weights[i]; */
    /* } */
    /* double* solution_vectors = cuda_double_copy_to_gpu(intermediate_solution, signal_elements); */
    /* free(intermediate_solution); */

    cublasStatus_t status;
    cublasHandle_t handle;
    int* cublas_error_info = (int*) malloc(sizeof(int));
    status = cublasCreate_v2(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        puts(cublas_get_error_string(status));
    }
    int* dev_info;
    cudaMalloc(&dev_info, sizeof(int) * batch_size);

    double** design_inter;
    design_inter = (double**) malloc(sizeof(double*) * batch_size);
    double** sol_inter;
    sol_inter = (double**) malloc(sizeof(double*) * batch_size);

    double** ls_weighted_design = convert_contigous_gpu_array_to_gpu_array_of_pointers(weighted_design_data, 
            design_matrix->rows, design_matrix->columns, batch_size, design_inter);
    double** ls_solution_vectors = convert_contigous_gpu_array_to_gpu_array_of_pointers(solution_vectors, 
            design_matrix->rows, 1, batch_size, sol_inter);

    status = cublasDgelsBatched(handle, CUBLAS_OP_N, design_matrix->rows, design_matrix->columns,
            1, ls_weighted_design, design_matrix->rows, ls_solution_vectors, design_matrix->rows, 
            cublas_error_info, dev_info, signals->rows);
    if (status != CUBLAS_STATUS_SUCCESS) {
        puts(cublas_get_error_string(status));
    }

    double** sol_array;
    sol_array = (double**) malloc(sizeof(double*) * batch_size);
    gpu_error_check(cudaMemcpy(sol_array, ls_solution_vectors, sizeof(double*) * batch_size, cudaMemcpyDeviceToHost));
    double* results;
    gpu_error_check(cudaMalloc(&results, sizeof(double) * design_matrix->columns * batch_size));
    int i, j, sol_offset;
    for(i = 0;i < batch_size;i++){
        sol_offset = i * design_matrix->columns ;
        for(j = 0;j < design_matrix->columns;j++){
            gpu_error_check(cudaMemcpy(results + sol_offset, sol_array[i], sizeof(double) * design_matrix->columns, cudaMemcpyDeviceToDevice));
        }
    }

    status = cublasDestroy_v2(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        puts(cublas_get_error_string(status));
    }

    return results;
}

extern "C"
double* cuda_decompose_tensors(double const* tensors_input, int tensor_input_elements, int number_of_tensors){
    double* tensors;
    cuda_double_allocate(&tensors, sizeof(double) * TENSOR_ELEMENTS * number_of_tensors);
    dim3 grid, block;
    grid.x = number_of_tensors;
    block.x = 1;
    block.y = 1;
    assemble_tensors<<<grid, block>>>(tensors_input, tensors, tensor_input_elements);
    double* gpu_eigendecomposition;
    int length_of_eigendecomposition = EIGENDECOMPOSITION_ELEMENTS * number_of_tensors;
    cuda_double_allocate(&gpu_eigendecomposition, sizeof(double) * length_of_eigendecomposition);
    eigendecomposition_kernel<<<grid, block>>>(tensors, gpu_eigendecomposition);
    double* eigendecomposition = cuda_double_return_from_gpu(gpu_eigendecomposition, length_of_eigendecomposition);
    free_cuda_memory(tensors);
    free_cuda_memory(gpu_eigendecomposition);
    return eigendecomposition;
}

extern "C"
matrix* process_matrix(matrix const* design_matrix){
    double* gpu_matrix_data = convert_matrix_to_fortran_and_load_to_gpu(design_matrix);
    matrix* processed_matrix = create_matrix(gpu_matrix_data, design_matrix->rows, design_matrix->columns);
    return processed_matrix;
}

extern "C"
void extract_eigendecompositions(double const* eigendecompositions, tensor** output, int number_of_tensors, double min_diffusivity){
    int i, j;
    for(i = 0; i < number_of_tensors;i++){
        double const* eigenvalue_pointer = eigendecompositions + (i * EIGENDECOMPOSITION_ELEMENTS);
        double const* eigenvector_pointer = eigendecompositions + ((i * EIGENDECOMPOSITION_ELEMENTS) + 3);
        double* eigenvalues = array_clone(eigenvalue_pointer, TENSOR_DIMENSIONS);
        double* eigenvectors = array_clone(eigenvector_pointer, TENSOR_ELEMENTS);        
        tensor* allocated_tensor = (tensor*) malloc(sizeof(tensor));
        output[i] = allocated_tensor;
        for(j = 0; j < TENSOR_DIMENSIONS; j++){
            if (eigenvalues[j] < min_diffusivity){
                eigenvalues[j] = min_diffusivity;
            }
        }
        output[i]->vals = eigenvalues;
        output[i]->vecs = create_matrix(eigenvectors, 3, 3);
    }
    return;
}

extern "C"
double* cuda_double_copy_to_gpu(double const* local_array, int array_length){
    double* cuda_array;
    gpu_error_check(cudaMalloc(&cuda_array, sizeof(double) * array_length));
    gpu_error_check(cudaMemcpy(cuda_array, local_array, sizeof(double) * array_length, cudaMemcpyHostToDevice));
    return cuda_array;
}
extern "C"
double* cuda_double_return_from_gpu(double const* cuda_array, int array_length){
    double* result_array = (double *) malloc(sizeof(double) * array_length);
    gpu_error_check(cudaMemcpy(result_array, cuda_array, sizeof(double) * array_length, cudaMemcpyDeviceToHost));
    return result_array;
}

extern "C"
void cuda_double_allocate(double** pointer, int pointer_length){
    gpu_error_check(cudaMalloc(pointer, pointer_length));
    gpu_error_check(cudaMemset(*pointer, 0, pointer_length));
}

extern "C"
void free_cuda_memory(double* pointer){
    gpu_error_check(cudaFree(pointer));
}

extern "C"
void free_matrix_with_cuda_pointer(matrix* gpu_matrix){
    free_cuda_memory(gpu_matrix->data);
    free(gpu_matrix);
}

extern "C"
double* cutoff_log_cuda(double const* input, double min_signal, int number_of_signals, int signal_length){
    int total_elements = number_of_signals * signal_length;
    double* device_array;
    gpu_error_check(cudaMalloc(&device_array, sizeof(double) * total_elements));
    gpu_error_check(cudaMemcpy(device_array, input, sizeof(double) * total_elements, cudaMemcpyDeviceToDevice))
    cutoff_log_kernel<<<number_of_signals, signal_length>>>(device_array, min_signal);
    return device_array;
}

extern "C"
double* exp_cuda(double const* input, int number_of_signals, int signal_length){
    int total_elements = number_of_signals * signal_length;
    double* device_array;
    gpu_error_check(cudaMalloc(&device_array, sizeof(double) * total_elements));
    gpu_error_check(cudaMemcpy(device_array, input, sizeof(double) * total_elements, cudaMemcpyDeviceToDevice))
    exp_kernel<<<number_of_signals, signal_length>>>(device_array);
    return device_array;
}

extern "C"
matrix* cuda_matrix_dot(matrix const* matrix1, matrix const* matrix2){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        puts(cublas_get_error_string(status));
    }
    double* gpu_output;
    gpu_error_check(cudaMalloc(&gpu_output, sizeof(double)* matrix1->rows * matrix2->columns));
    const double alpha = 1;
    const double beta = 0;
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix1->rows, matrix2->columns, matrix1->columns, 
            &alpha, matrix1->data, matrix1->rows, matrix2->data, matrix2->rows, &beta, gpu_output, matrix1->rows);
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        puts(cublas_get_error_string(status));
    }
    matrix* result_matrix = (matrix*) malloc(sizeof(matrix));
    double* result_matrix_data =  (double*) malloc(sizeof(double) * matrix1->rows * matrix2->columns);
    result_matrix->rows = matrix1->rows;
    result_matrix->columns = matrix2->columns;
    result_matrix->data = gpu_output;
    return result_matrix;
}
    
extern "C"
double* matrix_weighter (double const* gpu_matrix, double const* gpu_weights, int rows, int columns, int length, bool trans) {
    dim3 grid, block;
    grid.x = length;
    block.x = columns;
    block.y = rows;
    double* gpu_results;
    cuda_double_allocate(&gpu_results, sizeof(double) * rows * columns * length);
    if (false == trans){
        weighting_kernel<<<grid, block>>>(gpu_matrix, gpu_weights, gpu_results);
    }
    else {
        weighting_kernel_transposed<<<grid, block>>>(gpu_matrix, gpu_weights, gpu_results);
    }
    return gpu_results;
}

extern "C"
double* transpose_matrices(double const* matrices, int rows, int columns, int length){
    double* transposed = (double*) malloc(sizeof(double) * rows * columns * length);
    double* gpu_matrices = cuda_double_copy_to_gpu(matrices, rows * columns * length);
    double* gpu_transposed = cuda_double_copy_to_gpu(transposed, rows * columns * length);
    dim3 grid, block;
    grid.x = length;
    block.x = columns;
    block.y = rows;
    transpose_kernel<<<grid, block>>>(gpu_matrices, gpu_transposed);
    transposed = cuda_double_return_from_gpu(gpu_transposed, rows * columns * length);
    free_cuda_memory(gpu_matrices);
    free_cuda_memory(gpu_transposed);
    return transposed;
}

//Helper functions

/*Converts matrix to the data format fortran uses for CUBLAS and loads to GPU
  Returns pointer to array on GPU.*/
double* convert_matrix_to_fortran_and_load_to_gpu(matrix const* mat){
    int length = mat->rows * mat->columns;
    double* intermediate_matrix = (double*) malloc(sizeof(double) * length);
    int i, j, column_major_index;
    for (i = 0; i < mat->rows; i++ ) {
        for (j = 0; j < mat->columns; j++) {
            column_major_index  = IDX2C(i, j, mat->rows); 
            intermediate_matrix[column_major_index] = mat->data[i * mat->columns + j];
        }
    }
    double* gpu_array = cuda_double_copy_to_gpu(intermediate_matrix, length);
    free(intermediate_matrix);
    return gpu_array;
}

void get_matrix_from_gpu_and_convert_from_fortran(double const* gpu_pointer, matrix* mat){
    int length = mat->rows * mat->columns;
    double* intermediate_matrix = cuda_double_return_from_gpu(gpu_pointer, length);
    int i, j;
    for (i = 0; i < mat->rows; i++ ) {
        for (j = 0; j < mat->columns; j++) {
            mat->data[i * mat->columns + j] = intermediate_matrix[IDX2C(i, j, mat->rows)];
        }
    }
    free(intermediate_matrix);
}

/*Converts contigous array in gpu memory into array of pointers in gpu memory. Intermediate array must be an array of 
length (batch * sizeof(double*)). */
double** convert_contigous_gpu_array_to_gpu_array_of_pointers(double* arr, int m, int n, int batch, double** intermediate_array){
    int elements = m * n;
    int i, offset;
    for (i = 0; i < batch; i++){
        offset = i * elements;
        gpu_error_check(cudaMalloc(&intermediate_array[i], sizeof(double) * elements));
        gpu_error_check(cudaMemcpy(intermediate_array[i], arr + offset, sizeof(double) * elements, cudaMemcpyDeviceToDevice ));
    }
    double** gpu_array;
    gpu_error_check(cudaMalloc(&gpu_array, sizeof(double*) * batch));
    gpu_error_check(cudaMemcpy(gpu_array, intermediate_array, sizeof(double*) * batch, cudaMemcpyHostToDevice));
    return gpu_array;
}

//Frees array of pointers where array is on host and pointers are on the device.
void free_array_of_gpu_pointers(double** array, int batch){
    int i;
    for(i = 0;i < batch; i++){
        free_cuda_memory(array[i]);
    }
    free(array);
}


//Gets error and returns string based on it by the error returned.
const char* cublas_get_error_string(cublasStatus_t status){
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
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
__global__ void weighting_kernel (double const* matrices, double const* weights, double* results) {
    int matrix_grid_index = blockIdx.x * blockDim.x * blockDim.y;
    int block_index = blockDim.y * threadIdx.x + threadIdx.y;
    int matrix_index = matrix_grid_index + block_index;
    int weight_index = blockIdx.x * blockDim.y + threadIdx.y; 
    results[matrix_index] = matrices[block_index] * weights[weight_index];
}

//kernel for weighting a transposed matrix.
__global__ void weighting_kernel_transposed(double const* matrices, double const* weights, double* results) {
    int grid_index = blockIdx.x * blockDim.x * blockDim.y;
    int block_index = blockDim.y * threadIdx.x + threadIdx.y;
    int matrix_index = grid_index + block_index;
    int weighting_index = blockIdx.x * blockDim.x + threadIdx.x; 
    results[matrix_index] = matrices[block_index] * weights[weighting_index];
}

//kernel for transposing multiple matrices.
__global__ void transpose_kernel(double const* matrices, double* transposed) {
    int matrix_offset = blockIdx.x * blockDim.x * blockDim.y;
    int matrix_index = matrix_offset + blockDim.x * threadIdx.y + threadIdx.x;
    int transpose_index = matrix_offset + IDX2C(threadIdx.y, threadIdx.x, blockDim.y);
    transposed[transpose_index] = matrices[matrix_index];
}

//kernel for arranging tensors into symmetric matrix
__global__ void assemble_tensors(double const* tensor_input, double* tensors, int tensor_input_elements){
    int tensor_matrix_offset = blockIdx.x * TENSOR_DIMENSIONS * TENSOR_DIMENSIONS;
    int input_matrix_offset = blockIdx.x * tensor_input_elements;
    tensors[tensor_matrix_offset + 0] = tensor_input[input_matrix_offset + 0];
    tensors[tensor_matrix_offset + 1] = tensor_input[input_matrix_offset + 1];
    tensors[tensor_matrix_offset + 2] = tensor_input[input_matrix_offset + 3];
    tensors[tensor_matrix_offset + 3] = tensor_input[input_matrix_offset + 1];
    tensors[tensor_matrix_offset + 4] = tensor_input[input_matrix_offset + 2];
    tensors[tensor_matrix_offset + 5] = tensor_input[input_matrix_offset + 4];
    tensors[tensor_matrix_offset + 6] = tensor_input[input_matrix_offset + 3];
    tensors[tensor_matrix_offset + 7] = tensor_input[input_matrix_offset + 4];
    tensors[tensor_matrix_offset + 8] = tensor_input[input_matrix_offset + 5];
}

//kernel for calculating eigenvalues.
__global__ void eigendecomposition_kernel(double const* data, double* eigendecomposition){
    int matrix_offset = blockIdx.x * blockDim.x * TENSOR_DIMENSIONS * TENSOR_DIMENSIONS;
    int eigen_offset = blockIdx.x * blockDim.x * EIGENDECOMPOSITION_ELEMENTS;
    double A[3][3] = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    }; 
    deposit_data_segment_into_array(data, matrix_offset, A);
    double Q[3][3] = {
        {0, 0, 0},
        {0, 0, 0}, 
        {0, 0, 0}
    };
    double w[3] = {0, 0, 0};
    dsyevj3(A, Q, w);
    assemble_eigendecomposition(eigendecomposition, eigen_offset, Q, w);
}

//kernel to multiply two gpu arrays
__global__ void multiply_arrays(double* signals, double const* weights){
    signals[blockIdx.x * blockDim.x + threadIdx.x] *= weights[blockIdx.x * blockDim.x + threadIdx.x];
}

__global__ void create_array_of_pointers_kernel(double* arr, int m, int n, double** target){
    target[blockIdx.x] = arr + (blockIdx.x * m * n);
}

//device function of assembling eigendecomposition from respective blocks.
__device__ void assemble_eigendecomposition(double* eigendecomposition, int offset, 
        double Q[3][3], double w[3]){
    eigendecomposition[offset + 0] = w[0];
    eigendecomposition[offset + 1] = w[1];
    eigendecomposition[offset + 2] = w[2];
    eigendecomposition[offset + 3] = Q[0][0];
    eigendecomposition[offset + 4] = Q[0][1];
    eigendecomposition[offset + 5] = Q[0][2];
    eigendecomposition[offset + 6] = Q[1][0];
    eigendecomposition[offset + 7] = Q[1][1];
    eigendecomposition[offset + 8] = Q[1][2];
    eigendecomposition[offset + 9] = Q[2][0];
    eigendecomposition[offset + 10] = Q[2][1];
    eigendecomposition[offset + 11] = Q[2][2];
}

//device function to return tensor as array for eigendecompostion
__device__ void deposit_data_segment_into_array(double const* data, int matrix_offset, double A[3][3]){
    A[0][0] = data[matrix_offset];
    A[0][1] = data[matrix_offset + 1];
    A[0][2] = data[matrix_offset + 2];
    A[1][0] = data[matrix_offset + 3];
    A[1][1] = data[matrix_offset + 4];
    A[1][2] = data[matrix_offset + 5];
    A[2][0] = data[matrix_offset + 6];
    A[2][1] = data[matrix_offset + 7];
    A[2][2] = data[matrix_offset + 8];
}

// Jacobi algorithm for eigen decomposition. Implemented by Joachlm Kopp for his paper
// on 3x3 hermitian eigendecompositions. Copied here for use as a device function on a kernel.
__device__ int dsyevj3(double A[3][3], double Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using the Jacobi algorithm.
// The upper triangular part of A is destroyed during the calculation,
// the diagonal elements are read but not destroyed, and the lower
// triangular elements are not referenced at all.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error (no convergence)
// ----------------------------------------------------------------------------
{
  const int n = 3;
  double sd, so;                  // Sums of diagonal resp. off-diagonal elements
  double s, c, t;                 // sin(phi), cos(phi), tan(phi) and temporary storage
  double g, h, z, theta;          // More temporary storage
  double thresh;
  
  // Initialize Q to the identitity matrix
#ifndef EVALS_ONLY
  for (int i=0; i < n; i++)
  {
    Q[i][i] = 1.0;
    for (int j=0; j < i; j++)
      Q[i][j] = Q[j][i] = 0.0;
  }
#endif

  // Initialize w to diag(A)
  for (int i=0; i < n; i++)
    w[i] = A[i][i];

  // Calculate SQR(tr(A))  
  sd = 0.0;
  for (int i=0; i < n; i++)
    sd += fabs(w[i]);
  sd = SQR(sd);
 
  // Main iteration loop
  for (int nIter=0; nIter < 50; nIter++)
  {
    // Test for convergence 
    so = 0.0;
    for (int p=0; p < n; p++)
      for (int q=p+1; q < n; q++)
        so += fabs(A[p][q]);
    if (so == 0.0)
      return 0;

    if (nIter < 4)
      thresh = 0.2 * so / SQR(n);
    else
      thresh = 0.0;

    // Do sweep
    for (int p=0; p < n; p++)
      for (int q=p+1; q < n; q++)
      {
        g = 100.0 * fabs(A[p][q]);
        if (nIter > 4  &&  fabs(w[p]) + g == fabs(w[p])
                       &&  fabs(w[q]) + g == fabs(w[q]))
        {
          A[p][q] = 0.0;
        }
        else if (fabs(A[p][q]) > thresh)
        {
          // Calculate Jacobi transformation
          h = w[q] - w[p];
          if (fabs(h) + g == fabs(h))
          {
            t = A[p][q] / h;
          }
          else
          {
            theta = 0.5 * h / A[p][q];
            if (theta < 0.0)
              t = -1.0 / (sqrt(1.0 + SQR(theta)) - theta);
            else
              t = 1.0 / (sqrt(1.0 + SQR(theta)) + theta);
          }
          c = 1.0/sqrt(1.0 + SQR(t));
          s = t * c;
          z = t * A[p][q];

          // Apply Jacobi transformation
          A[p][q] = 0.0;
          w[p] -= z;
          w[q] += z;
          for (int r=0; r < p; r++)
          {
            t = A[r][p];
            A[r][p] = c*t - s*A[r][q];
            A[r][q] = s*t + c*A[r][q];
          }
          for (int r=p+1; r < q; r++)
          {
            t = A[p][r];
            A[p][r] = c*t - s*A[r][q];
            A[r][q] = s*t + c*A[r][q];
          }
          for (int r=q+1; r < n; r++)
          {
            t = A[p][r];
            A[p][r] = c*t - s*A[q][r];
            A[q][r] = s*t + c*A[q][r];
          }

          // Update eigenvectors
#ifndef EVALS_ONLY          
          for (int r=0; r < n; r++)
          {
            t = Q[r][p];
            Q[r][p] = c*t - s*Q[r][q];
            Q[r][q] = s*t + c*Q[r][q];
          }
#endif
        }
      }
  }

  return -1;
}

