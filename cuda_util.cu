#include <stdlib.h>
#include <cublas_v2.h>
extern "C" {
#include "cuda_util.h"
}
#include "BatchedSolver/solve.h"

#define IDX2C(i, j, ld) ((j)*(ld)+(i))
#define SQR(x)      ((x)*(x))                        // x^2 

#define WARP_SIZE 16

#define TENSOR_DIMENSIONS 3
#define TENSOR_INPUT_ELEMENTS 6
#define TENSOR_ELEMENTS 9
#define EIGENDECOMPOSITION_ELEMENTS 12


//Helper function declarations
double* convert_matrix_to_fortran_and_load_to_gpu(matrix const* mat);
void get_matrix_from_gpu_and_convert_from_fortran(double const* gpu_pointer, matrix* mat);

//Kernel declarations
__global__ void cutoff_log_kernel(double* device_array, double min_signal);
__global__ void exp_kernel(double* cuda_array);
__global__ void weighting_kernel (double* matrices, double* weights, double* results); 
__global__ void weighting_kernel_transposed(double* matrices, double* weights, double* results); 
__global__ void transpose_kernel(double const* matrices, double* transposed);
__global__ void assemble_tensors(double const* tensor_input, double* tensors);
__global__ void eigendecomposition_kernel(double const* data, double* eigendecomposition);

//device functions
__device__ void assemble_eigendecomposition(double* eigendecomposition, double* offset, 
        double Q[3][3], double w[3]);
__device__ int dsyevj3(double A[3][3], double Q[3][3], double w[3]);

extern "C"
matrix* process_signal(matrix const* signal, double min_signal){
    double* signal_data = array_clone(signal->data, signal->rows * signal->columns);
    int signal_length = signal->rows * signal->columns;
    double* kernel_results = cutoff_log_cuda(signal_data, min_signal, signal_length);
    double* processed_signal_data = cuda_double_copy_to_gpu(kernel_results, signal_length);
    matrix* processed_signal = create_matrix(processed_signal_data, signal->rows, signal->columns);
    free(signal_data);
    free_cuda_memory(kernel_results);
    return processed_signal;
}

extern "C"
matrix* generate_weights(matrix const* ols_fit_matrix, matrix const* signal){
    matrix* weights = cuda_matrix_dot(ols_fit_matrix, signal);
    double* gpu_weights_data = cuda_double_copy_to_gpu(weights->data, weights->rows * weights->columns);
    matrix* gpu_weights= create_matrix( gpu_weights_data, weights->rows, weights->columns);
    free_matrix(weights);
    return gpu_weights;
}

extern "C"
double* cuda_fitter(matrix const* design_matrix, matrix const* column_major_weights, matrix const* signal){
    double* weighted_design_data = matrix_weighter(design_matrix->data, column_weights->data, design_matrix->rows, 
            design_matrix->columns, column_major_weights->rows, true);
    double* solution_vectors;
    int signal_elements = signal->rows * signal->columns;
    cuda_double_allocate(solution_vectors, signal_elements);
    int solver_status = dsolve_batch(weighted_design_data, signal->data, solution_vectors, 
            signal->columns, signal->rows);
    if ( 0 > solver_status) {
        fputs("Batched solver failed to run correctly, program will fail", stderr);
    }
    free_cuda_memory(solution_vectors);
    return solution_vectors;
}

extern "C"
double* cuda_decompose_tensors(double const* tensors_input, int number_of_tensors){
    double* tensors;
    cuda_double_allocate(tensors, TENSOR_ELEMENTS * number_of_tensors);
    dim3 grid, block;
    grid.x = number_of_tensors;
    block.x = 1;
    block.y = 1;
    assemble_tensors<<<grid, block>>>(tensors_input, tensors);
    double* gpu_eigendecomposition;
    int length_of_eigendecomposition = EIGENDECOMPOSITION_ELEMENTS * number_of_tensors;
    cuda_double_allocate(gpu_eigendecomposition, length_of_eigendecomposition);
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
void extract_eigendecompositions(double const* eigendecompositions, tensor** output, int number_of_tensors){
    int i;
    for(i = 0; i < number_of_tensors;i++){
        double* eigenvalues = array_clone((double const*)eigendecompositions[i * EIGENDECOMPOSITION_ELEMENTS], TENSOR_DIMENSIONS);
        double* eigenvectors = array_clone((double const *)eigendecompositions[(i * EIGENDECOMPOSITION_ELEMENTS) + 3],
            TENSOR_ELEMENTS);        
        output[i]->vals = eigenvalues;
        output[i]->vecs = eigenvectors;
    }
    return;
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
    cudaMemset(&pointer, 0, pointer_length);
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

//kernel for arranging tensors into symmetric matrix
__global__ void assemble_tensors(double const* tensor_input, double* tensors){
    int tensor_matrix_offset = blockIdx.x * TENSOR_DIMENSIONS * TENSOR_DIMENSIONS;
    int input_matrix_offset = blockIdx.x * TENSOR_INPUT_ELEMENTS;
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
    int matrix_offset = blockIdx.x * blockDim.x * TENSOR_DIMENSIONS;
    int eigen_offset = blockIdx.x * blockDim.x * EIGENDECOMPOSITION_ELEMENTS;
    double A[3][3] = deposit_data_segment_into_array(data, matrix_offset);
    double Q[3][3] = {0};
    double w[3] = {0};
    dsyevj3(A, Q, w);
    assemble_eigendecomposition(eigendecomposition, eigen_offset, A, Q, w);
}

//device function of assembling eigendecomposition from respective blocks.
__device__ void assemble_eigendecomposition(double* eigendecomposition, double* offset, 
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

