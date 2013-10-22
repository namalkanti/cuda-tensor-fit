#include "cuda_util.h"
#include "fit_tensor.h"

void fit_complete_signal(matrix* ols_fit, matrix* design_matrix, matrix* signal, double min_signal, double min_diffusivity, tensor** tensor_output) {
    int signal_length = signal->rows * signal->length;
    signal->data = cutoff_log_cuda(signal->data, min_signal, signal_length);
    matrix* weights = cuda_matrix_dot(ols_fit, signal); 
    exp_kernel(weights->data);
    double* tensors = cuda_fitter(design_matrix, weights, signal, signal_length); 
    cuda_decompose_tensors(tensors, tensor_output);
}

