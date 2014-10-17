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
double* convert_matrix_to_fortran_and_load_to_gpu(matrix const* mat);
void get_matrix_from_gpu_and_convert_from_fortran(double const* gpu_pointer, matrix* mat);
const char* cublas_get_error_string(cublasStatus_t status);

//Kernel declarations
__global__ void cutoff_log_kernel(double* device_array, double min_signal);
__global__ void exp_kernel(double* cuda_array);
__global__ void weighting_kernel (double const* matrices, double const* weights, double* results); 
__global__ void weighting_kernel_transposed(double const* matrices, double const* weights, double* results); 
__global__ void transpose_kernel(double const* matrices, double* transposed);
__global__ void assemble_tensors(double const* tensor_input, double* tensors);
__global__ void eigendecomposition_kernel(double const* data, double* eigendecomposition);

//device functions
__device__ void assemble_eigendecomposition(double* eigendecomposition, int offset, double Q[3][3], double w[3]);
__device__ void deposit_data_segment_into_array(double const* data, int offset, double A[3][3]);
__device__ int dsyevj3(double A[3][3], double Q[3][3], double w[3]);

extern "C"
matrix* process_signal(matrix const* signal, double min_signal){
    double* signal_data = array_clone(signal->data, signal->rows * signal->columns);
    int signal_length = signal->rows * signal->columns;
    double* kernel_results = cutoff_log_cuda(signal_data, min_signal, signal_length);
    double* processed_signal_data = cuda_double_copy_to_gpu(kernel_results, signal_length);
    matrix* processed_signal = create_matrix(processed_signal_data, signal->rows, signal->columns);
    free(signal_data);
    free(kernel_results);
    return processed_signal;
}

extern "C"
matrix* generate_weights(matrix const* ols_fit_matrix, matrix const* signal){
    matrix* weights = cuda_matrix_dot(ols_fit_matrix, signal);
    double* exp_weights = exp_cuda(weights->data, weights->rows * weights->columns);
    free(weights->data);
    weights->data = exp_weights;
    double* gpu_weights_data = cuda_double_copy_to_gpu(weights->data, weights->rows * weights->columns);
    matrix* gpu_weights= create_matrix(gpu_weights_data, weights->rows, weights->columns);
    free_matrix(weights);
    return gpu_weights;
}

extern "C"
double* cuda_fitter(matrix const* design_matrix, matrix const* column_major_weights, matrix const* signals){
    //Will not transpose matrix weighting because design matrix is column major already
    double* weighted_design_data = matrix_weighter(design_matrix->data, column_major_weights->data, 
            design_matrix->rows, design_matrix->columns, column_major_weights->rows, false);
    double* solution_vectors;
    int signal_elements = signals->rows * signals->columns;
    cuda_double_allocate(&solution_vectors, sizeof(double) *signal_elements);
    cublasState_t status;
    cublasHandle_t handle;
    int* cublas_error_info = 0;;
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        puts(cublas_get_error_string(status));
    }
    status = cublasDgelsBatched(handle, CUBLAS_OP_N, design_matrix->rows, design_matrix->columns,
            design_matrix->rows, weighted_design_data, design_matrix->rows, signals, design_matrix->rows, 
            cublas_error_info, NULL, signals->columns);
    /* int solver_status = dsolve_batch(weighted_design_data, signals->data, solution_vectors, */ 
    /*         signals->columns, signals->rows); */
    if (status != CUBLAS_STATUS_SUCCESS) {
        puts(cublas_get_error_string(status));
    }
    free_cuda_memory(weighted_design_data);
    return solution_vectors;
}

extern "C"
double* cuda_decompose_tensors(double const* tensors_input, int number_of_tensors){
    double* tensors;
    cuda_double_allocate(&tensors, sizeof(double) * TENSOR_ELEMENTS * number_of_tensors);
    dim3 grid, block;
    grid.x = number_of_tensors;
    block.x = 1;
    block.y = 1;
    assemble_tensors<<<grid, block>>>(tensors_input, tensors);
    double* debug_tensors = cuda_double_return_from_gpu(tensors, TENSOR_ELEMENTS * number_of_tensors);
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
void extract_eigendecompositions(double const* eigendecompositions, tensor** output, int number_of_tensors){
    int i;
    for(i = 0; i < number_of_tensors;i++){
        double const* eigenvalue_pointer = eigendecompositions + (i * EIGENDECOMPOSITION_ELEMENTS);
        double const* eigenvector_pointer = eigendecompositions + ((i * EIGENDECOMPOSITION_ELEMENTS) + 3);
        double* eigenvalues = array_clone(eigenvalue_pointer, TENSOR_DIMENSIONS);
        double* eigenvectors = array_clone(eigenvector_pointer, TENSOR_ELEMENTS);        
        tensor* allocated_tensor = (tensor*) malloc(sizeof(tensor));
        output[i] = allocated_tensor;
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
double* cutoff_log_cuda(double const* input, double min_signal, int array_length){
    padded_array* padded_arr = pad_array(array_clone(input, array_length), array_length, WARP_SIZE);
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
        puts(cublas_get_error_string(status));
    }
    double* gpu_array1 = convert_matrix_to_fortran_and_load_to_gpu(matrix1);
    double* gpu_array2 = convert_matrix_to_fortran_and_load_to_gpu(matrix2);
    double* gpu_output;
    gpu_error_check(cudaMalloc(&gpu_output, sizeof(double)* matrix1->rows * matrix2->columns));
    const double alpha = 1;
    const double beta = 0;
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix1->rows, matrix2->columns, matrix1->columns, 
            &alpha, gpu_array1, matrix1->rows, gpu_array2, matrix2->rows, &beta, gpu_output, matrix1->rows);
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        puts(cublas_get_error_string(status));
    }
    matrix* result_matrix = (matrix*) malloc(sizeof(matrix));
    double* result_matrix_data =  (double*) malloc(sizeof(double) * matrix1->rows * matrix2->columns);
    result_matrix->rows = matrix1->rows;
    result_matrix->columns = matrix2->columns;
    result_matrix->data = result_matrix_data;
    get_matrix_from_gpu_and_convert_from_fortran(gpu_output, result_matrix);
    gpu_error_check(cudaFree(gpu_array1));
    gpu_error_check(cudaFree(gpu_array2));
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

extern "C"
double* dot_matrices(double const* matrix_batch_one, int rows, double const* matrix_batch_two, int columns,
        int k, int length){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        puts(cublas_get_error_string(status));
    }
    double* transposed_batch1 = transpose_matrices(matrix_batch_one, rows, k, length);
    double* transposed_batch2 = transpose_matrices(matrix_batch_two, k, columns, length);
    double* gpu_array1 = cuda_double_copy_to_gpu(transposed_batch1, rows * k * length);
    double* gpu_array2 = cuda_double_copy_to_gpu(transposed_batch2, k *  columns * length);
    double* gpu_output;
    cudaMalloc(&gpu_output, sizeof(double)* rows * columns * length);
    const double alpha = 1;
    const double beta = 0;
    status = cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, columns, 
            k, &alpha, (const double**) &gpu_array1, rows, (const double**) &gpu_array2, k, &beta, 
            &gpu_output, rows, length);
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        puts(cublas_get_error_string(status));
    }
    double *transposed_results, *results;
    transposed_results = cuda_double_return_from_gpu(gpu_output, rows * columns * length);
    results = transpose_matrices(transposed_results, rows, columns, length);
    free_cuda_memory(gpu_array1);
    free_cuda_memory(gpu_array2);
    free_cuda_memory(gpu_output);
    free(transposed_batch1);
    free(transposed_batch2);
    free(transposed_results);
    cublasDestroy(handle);
    return results;

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

/*Converts matrix from the format fortran uses for CUBLAS after retrieving from GPU
  Will free gpu_pointer.
  Populates a matrix object passed in.*/
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
    int block_index = blockDim.y * threadIdx.y + threadIdx.x;
    int matrix_index = matrix_grid_index + block_index;
    int weight_index = blockIdx.x * blockDim.x + threadIdx.x; 
    results[matrix_index] = matrices[block_index] * weights[weight_index];
}

//kernel for weighting a transposed matrix.
__global__ void weighting_kernel_transposed(double const* matrices, double const* weights, double* results) {
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
    int transpose_index = matrix_offset + IDX2C(threadIdx.y, threadIdx.x, blockDim.y);
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

