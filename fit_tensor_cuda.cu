#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fit_tensor.h"

//clones double array and copies to gpu
double* cuda_double_copy(double* arr, size_t len){
    double* carr;
    cudaMalloc(&carr, sizeof(double) * len);
    cudaMemcpy(carr, arr, sizeof(double) * len, cudaMemcpyHostToDevice);
    return carr;
}

//clones double array and copies to host 
double* cuda_double_return(double* carr, size_t len){
    double* arr = (double*) malloc(sizeof(double) * len);
    cudaMemcpy(arr, carr, sizeof(double) * len, cudaMemcpyDeviceToHost);
    return arr;
}

//allocates space for a double array on the device
void cudal_double_alloc(double* ptr, int len){
    cudaMalloc(&ptr, len);
}

//frees double device memory
void free_cuda(double* ptr){
    cudaFree(ptr);
}

//kernel to take entire array and run cutoff log function
__global__ void cutoff_log_kernel(double* input, double* output, double min_signal, size_t len){
    int tid = blockIdx.x;
    if (input[tid] < min_signal){
        output[tid] = log(min_signal);
    }
    else{
        output[tid] = log(input[tid]);
    }
}

//kernel to take entire array and exp it
__global__ void exp_kernel(double* input, double* output, size_t len){
    int tid = blockIdx.x;
    output[tid] = pow(M_E, input[tid]);
}

//function that take in a complete signal matrix and fits it, cuda version
void fit_complete_signal(matrix* ols_fit, matrix* design_matrix, matrix* signal, double min_signal, double min_diffusivity, tensor** tensor_output){
    return;
}
