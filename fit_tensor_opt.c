#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "fit_tensor.h"

//Function to fit each individual signal row
tensor* signal_fit(matrix* ols_fit, matrix* design_matrix, double* sig, double min_signal, double min_diffusivity, size_t signal_length){
    cutoff_log(sig, min_signal, signal_length);
    matrix signal_matrix = {sig, signal_length, 1};
    matrix* ols_signal_dot = matrix_dot(ols_fit, &signal_matrix);
    double* weights = exp_array(ols_signal_dot->data, signal_length);
    double* D = fitter(design_matrix, weights, sig, signal_length);
    matrix* tensor_matrix = tensor_lower_triangular(D);
    tensor* signal_tensor = decompose_tensor(tensor_matrix, min_diffusivity);
    free_matrix(ols_signal_dot);
    free(weights);
    free(D);
    free_matrix(tensor_matrix);
    return signal_tensor;
}

//Function that take in a complete signal matrix and fits it
void fit_complete_signal(matrix* ols_fit, matrix* design_matrix, matrix* signal, double min_signal, double min_diffusivity, tensor** tensor_output){
    int i;
#pragma omp parallel for
    for(i = 0; i < signal->rows; i++){
        double* sig = malloc(sizeof(double) * signal->columns);
        int j;
        for(j = 0; j < signal->columns;j++){
            sig[j] = signal->data[i * signal->columns + j];
        } 
        tensor_output[i] = signal_fit(ols_fit, design_matrix, sig, min_signal, min_diffusivity, signal->columns);
        free(sig);
    }
}
