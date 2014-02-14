#include "cuda_util.h"
#include "fit_tensor.h"

void fit_complete_signal(matrix* ols_fit, matrix* design_matrix, matrix* signal, double min_signal, 
        double min_diffusivity, tensor** tensor_output) {
    matrix* processed_signal_gpu = process_signal(signal, min_signal);
    matrix* column_major_weights_gpu = generate_weights(ols_fit, signal);
    matrix* column_major_design_matrix_gpu = process_matrix(design_matrix);
    double* tensors_gpu = cuda_fitter(column_major_design_matrix_gpu, column_major_weights, 
            processed_signal_gpu);
    double* padded_eigendecompositions_gpu = cuda_decompose_tensors(tensors_gpu);
    extract_eigendecompositions(padded_eigendecompositions_gpu, tensor_output);
}
    int padded_signal_length = round_to_power_of_two(signal->rows);
    int padded_number_of_signals = round_to_power_of_two(signal->columns);
    int padded_signal_elements = padded_signal_length * padded_number_of_signals;
    double* gpu_signal = prepare_signal_and_load_to_gpu();
    double* gpu_logged_signal = cutoff_log_cuda(gpu_signal, min_signal, padded_signal_elements);
    matrix* weights = cuda_matrix_dot(ols_fit, signal); 
    matrix* column_major_weights = transpose(weights);
    exp_kernel(columns_major_weights->data);
    matrix* column_major_design_matrix = transpose(design_matrix);
    double* tensors = cuda_fitter(column_major_design_matrix, column_major_weights, gpu_logged_signal, 
            padded_signal_length, padded_number_of_signals); 
    cuda_decompose_tensors(tensors, tensor_output);

