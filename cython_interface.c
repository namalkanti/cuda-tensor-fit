#include "cython_interface.h"
#define EIGENVALUES 3
#define EIGENVECTORS 9
#define EIGENELEMENTS 12

void python_to_c(double* ols_fit_data, int ols_rows, int ols_columns, 
                double* design_matrix_data, int design_rows, int design_columns,
                double* signal_data, int signals, int signal_elements,
                double min_signal, double min_diffusivity, double* output){

    matrix* ols_fit = create_matrix(ols_fit_data, ols_rows, ols_columns);
    matrix* design_matrix = create_matrix(design_matrix_data, design_rows, design_columns);
    matrix* signal = create_matrix(signal_data, signals, signal_elements);

    tensor* tensor_output[signals]; 


    fit_complete_signal(ols_fit, design_matrix, signal, min_signal, min_diffusivity, tensor_output);

    int i, j;
    for (i = 0; i < signals; i++){
        for (j = 0; j < EIGENVALUES; j++){
            output[i * EIGENELEMENTS + j] = tensor_output[i]->vals[j];
        }
        for ( j = 0; j < EIGENVECTORS; j++){
            output[i * EIGENELEMENTS + EIGENVALUES + j] = tensor_output[i]->vecs->data[j];
        }
    }
}
