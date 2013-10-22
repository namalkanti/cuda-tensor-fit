#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32

//clones double array and copies to gpu
double* cuda_double_copy_to_gpu(double* local_array, int array_length);

//clones double array and copies to host 
double* cuda_double_return_from_gpu(double* cuda_array, int array_length);

//allocates space for a double array on the device
void cuda_double_allocate(double* pointer, int pointer_length);

//frees double device memory
void free_cuda_memory(double* pointer);

//kernel to take entire array and run cutoff log function
double* cutoff_log_cuda(double* input, double min_signal, int array_length);

//kernel to take entire array and exp it
double* exp_cuda(double* input, int array_length);

//Wrapper for CUBLASS Dgemm call
matrix* cuda_matrix_dot(matrix* matrix1, matrix* matrix2);

//Wrapper function to weight and fit the data
double* cuda_fitter(matrix* design_matrix, matrix* weights, signal, signal_length);

//Decomposes tensors and places them inside second argument.
void decompose_tensors(double* tensors, tensor** tensor_output);
