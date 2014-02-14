#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include "opt_util.h"

void cutoff_log(double* signal, double min_signal, int n){
    int i;
    for (i = 0; i < n; i++){
        if (signal[i] < min_signal){
            signal[i] = min_signal;
        }
        signal[i] = log(signal[i]);
    }
}

double* exp_array(double const* input, int n){
    double* output = (double*) malloc(sizeof(double) * n);
    int i;
    for(i = 0; i < n; i++){
        output[i] = pow(M_E, input[i]);
    }
    return output;
}

tensor* decompose_tensor_matrix(matrix const* tensor_matrix, const double min_diffusitivity){
    gsl_vector* evals = gsl_vector_alloc(3);
    gsl_matrix* evecs = gsl_matrix_alloc(3, 3);
    gsl_eigen_symmv_workspace* w = gsl_eigen_symmv_alloc(4);
    gsl_matrix* a = to_gsl(tensor_matrix);
    gsl_eigen_symmv(a, evals, evecs, w);
    gsl_eigen_symmv_sort(evals, evecs, GSL_EIGEN_SORT_VAL_DESC);
    tensor* tensor_output = malloc(sizeof(tensor));    
    double* vals = malloc(sizeof(double) * 3);
    matrix* vecs = to_matrix(evecs);
    int i;
    double val;
    for (i = 0; i < 3; i++){
        val = gsl_vector_get(evals, i);
        if (val < min_diffusitivity){
            val = min_diffusitivity;
        }
        vals[i] = val;
    }
    tensor_output->vals = vals;
    tensor_output->vecs = vecs;
    gsl_eigen_symmv_free(w);
    gsl_vector_free(evals);
    gsl_matrix_free(evecs);
    gsl_matrix_free(a);
    return tensor_output;
}

double* fit_matrix(matrix const* design, double const* weights, double const* signal, int sig_size){
    gsl_vector* signal_gsl = gsl_vector_alloc(sig_size);
    double* signal_clone = array_clone(signal, sig_size);
    matrix signal_mat = {signal_clone, sig_size, 1};
    int i;
    for (i = 0; i < sig_size; i++){
        signal_mat.data[i] = signal[i] * weights[i]; 
        gsl_vector_set(signal_gsl, i, signal[i] * weights[i]);
    }
    matrix* weighted_design = scale_matrix(design, weights, 1);
    gsl_matrix* weighted_design_gsl = to_gsl(weighted_design);
    gsl_matrix* trans_gsl = gsl_matrix_alloc(weighted_design_gsl->size2, weighted_design_gsl->size1);	
    gsl_matrix_transpose_memcpy(trans_gsl, weighted_design_gsl);
    matrix* trans = to_matrix(trans_gsl);
    matrix* fit = matrix_dot(trans, weighted_design);
    matrix* inter_signal = matrix_dot(trans, &signal_mat);
    gsl_vector* inter_sig_gsl = gsl_vector_alloc(inter_signal->rows);
    for (i = 0; i < inter_sig_gsl->size; i++){
        gsl_vector_set(inter_sig_gsl, i, inter_signal->data[i]);
    }
    gsl_matrix* fit_gsl = to_gsl(fit);
    int signum;
    gsl_permutation* p = gsl_permutation_alloc(fit_gsl->size1);
    gsl_linalg_LU_decomp(fit_gsl, p, &signum);
    gsl_linalg_LU_svx(fit_gsl, p, inter_sig_gsl);
    double* output = array_clone(inter_sig_gsl->data, inter_sig_gsl->size);
    gsl_vector_free(signal_gsl);
    free(signal_clone);
    free_matrix(weighted_design);
    gsl_matrix_free(weighted_design_gsl);
    gsl_matrix_free(trans_gsl);
    free_matrix(trans);
    free_matrix(fit);
    free_matrix(inter_signal);
    gsl_vector_free(inter_sig_gsl);
    gsl_matrix_free(fit_gsl);
    gsl_permutation_free(p);
    return output;
}
