#include <stdio.h>
#include <stdlib.h>

//clones double array and copies to gpu
double* cuda_double_copy(double* arr, int len);

//clones double array and copies to host 
double* cuda_double_return(double* carr, int len);

//allocates space for a double array on the device
void cuda_double_alloc(double* ptr, int len);

//frees double device memory
void free_cuda(double* ptr);

//kernel to take entire array and run cutoff log function
void cutoff_log_cuda(double* input, double* output, double min_signal, int block_grid_rows);

//kernel to take entire array and exp it
void exp_cuda(double* input, double* output, int block_grid_rows);
