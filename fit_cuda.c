#include "cuda_util.h"

void fit_complete_signal(matrix* ols_fit, matrix* design_matrix, matrix* signal, double min_signal, 
        double min_diffusivity, tensor** tensor_output) {
    int number_of_signals = signal->rows;
    int signal_elements = signal->columns;
    matrix* processed_signal_gpu = process_signal(signal, min_signal);
    signal->data = cuda_double_return_from_gpu(process_matrix->data, number_of_signals * signal_elements);
    matrix* column_major_weights_gpu = generate_weights(ols_fit, signal);
    matrix* column_major_design_matrix_gpu = process_matrix(design_matrix);
    double* tensors_gpu = cuda_fitter(column_major_design_matrix_gpu, column_major_weights_gpu, 
            processed_signal_gpu);
    double* padded_eigendecompositions_gpu = cuda_decompose_tensors(tensors_gpu, number_of_signals);
    extract_eigendecompositions(padded_eigendecompositions_gpu, tensor_output, signal_elements);
}
