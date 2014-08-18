#include "cuda_util.h"

void fit_complete_signal(matrix* ols_fit, matrix* design_matrix, matrix* signal, double min_signal, 
        double min_diffusivity, tensor** tensor_output) {
    matrix* processed_signal_gpu = process_signal(signal, min_signal);
    // Need to change to processed_signal, also need to check gpu pointer or no
    matrix* column_major_weights_gpu = generate_weights(ols_fit, signal);
    matrix* column_major_design_matrix_gpu = process_matrix(design_matrix);
    double* tensors_gpu = cuda_fitter(column_major_design_matrix_gpu, column_major_weights, 
            processed_signal_gpu);
    double* padded_eigendecompositions_gpu = cuda_decompose_tensors(tensors_gpu);
    extract_eigendecompositions(padded_eigendecompositions_gpu, tensor_output, signal->columns);
}
