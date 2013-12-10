#include <stdio.h>
#include <stdlib.h>
#include "fit_tensor.h"

#define WARP_SIZE 32

//Clones double array and copies to gpu
double* cuda_double_copy_to_gpu(double const* local_array, int array_length);

//Clones double array and copies to host 
double* cuda_double_return_from_gpu(double const* cuda_array, int array_length);

//Allocates space for a double array on the device
void cuda_double_allocate(double* pointer, int pointer_length);

//Frees double device memory
void free_cuda_memory(double const* pointer);

//Kernel to take entire array and run cutoff log function
double* cutoff_log_cuda(double const* input, double min_signal, int array_length);

//Kernel to take entire array and exp it
double* exp_cuda(double const* input, int array_length);

//Wrapper for CUBLAS Dgemm call
matrix* cuda_matrix_dot(matrix const* matrix1, matrix const* matrix2);

//Function weights matrix according to trans argument.
void matrix_weighter (double* matrices, double const* weights, int rows, int columns, int length, bool trans);

//Wrapper function to weight and fit the data
double* cuda_fitter(matrix const* design_matrix, matrix const* weights, double const* signal, int signal_length);

//Decomposes tensors and places them inside second argument.
void decompose_tensors(double const* tensors, tensor** tensor_output);
