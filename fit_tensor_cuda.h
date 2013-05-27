#include <stdio.h>
#include <stdlib.h>
#include "fit_tensor.h"

//clones double array and copies to gpu
double* cuda_double_copy(double* arr, size_t len);

//clones double array and copies to host 
double* cuda_double_return(double* carr, size_t len);

//allocates space for a double array on the device
void cudal_double_alloc(double* ptr, int len);

//frees double device memory
void free_cuda(double* ptr);

//kernel to take entire array and run cutoff log function
__global__ void cutoff_log_kernel(double* input, double* output, double min_signal, size_t len);

//kernel to take entire array and exp it
__global__ void exp_kernel(double* input, double* output, size_t len);
