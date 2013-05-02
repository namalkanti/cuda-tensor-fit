#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include "data.h"

//emulates numpy's maximum function and log function combined for efficiency.
//iterates through array and if value is less than min signal, replaces with minimum value.
//also takes logarithm of every value.
void cutoff_log(double* signal, double min_signal, size_t n){
    int i;
    for (i = 0; i < n; i++){
        if (signal[i] < min_signal){
            signal[i] = min_signal;
        }
        signal[i] = log(signal[i]);
    }
}

//raises every e to the power of every element in the input array and outputs new array
//pointer returned is allocated on heap, free memory when done using it
double* exp_array(double* input, size_t n){
    double* output = (double*) malloc(sizeof(double) * n);
    int i;
    for(i = 0; i < n; i++){
        output[i] = pow(M_E, input[i]);
    }
    return output;
}

//function takes input array and matches each index to a certain position in a 3 x 3 matrix
//only uses first six elements of input array. will fail if less than six are provided.
matrix* tensor_lower_triangular(double* input){
    double* output = (double*) malloc(sizeof(double) * 9);
    output[0] = input[0];
    output[1] = input[1];
    output[2] = input[3];
    output[3] = input[1];
    output[4] = input[2];
    output[5] = input[4];
    output[6] = input[3];
    output[7] = input[4];
    output[8] = input[5];
    matrix* output_mat = (matrix*) malloc(sizeof(matrix));
    output_mat->data = output;
    output_mat->rows = 3;
    output_mat->columns = 3;
    return output_mat;
}

//takes in a matrix and and an array
//multiplies each matrix row element by it's corresponding array element and repeats for all rows if transpose flag is 0
//if transpose flag is 1 will multiply by each column instead
matrix* matrix_scale(matrix* input_matrix, double* vec, int trans){
    double* output = (double*) malloc(sizeof(double) * (input_matrix->rows * input_matrix->columns));
    int i, j;
    int columns = input_matrix->columns;
    switch(trans)
    {
        case 0:
            for(i = 0; i < input_matrix->rows; i++){
                for(j = 0; j < columns; j++){
                    output[ i * columns + j] = input_matrix->data[i * columns + j] * vec[j];
                }
            }
            break;
        case 1:
            for(i = 0; i < input_matrix->rows; i++){
                for(j = 0; j < columns; j++){
                    output[i * columns + j] = vec[i] * input_matrix->data[i * columns + j];
                }
            }
            break;
    }
    matrix* output_mat = (matrix*) malloc(sizeof(matrix));
    output_mat->data = output;
    output_mat->rows = input_matrix->rows;
    output_mat->columns = input_matrix->columns;
    return output_mat;
}

//returns gsl matrix for interfacing with gsl library for blas and lapack
//meant to be used a helper function
gsl_matrix* to_gsl(matrix* mat){
    gsl_matrix* output = gsl_matrix_alloc(mat->rows, mat->columns);
    int i, j;
    for(i = 0; i < mat->rows; i++){
        for( j = 0; j < mat->columns; j++){
            gsl_matrix_set(output, i, j, mat->data[i * mat->columns + j]);
        }
    }
    return output;
}

//returns matrix from gsl matrix.
matrix* to_matrix(gsl_matrix* gsl_mat){
    int i, j;
    double* output_data = (double*) malloc(sizeof(double) * ((int)gsl_mat->size1) * ((int)gsl_mat->size2));
    matrix* output = malloc(sizeof(matrix));
    output->rows = (int) gsl_mat-> size1;
    output->columns = (int) gsl_mat-> size2;
    for(i = 0; i < output->rows; i++){
        for( j = 0; j < output->columns; j++){
            output_data[i * output->columns + j] = gsl_matrix_get(gsl_mat, i, j);
        }
    }
    output->data = output_data;
    return output;
}

//function to take two matrices, dot them, and return the result
matrix* matrix_dot(matrix* a, matrix* b){
    double* c_data = malloc(sizeof(double) * a->rows * b->columns); 
    matrix* c = malloc(sizeof(matrix));
    c->rows = a->rows;
    c->columns = b->columns;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a->rows, b->columns, a->columns, 1.0, a->data, a->columns, b->data, b->columns, 0.0, c_data, c->columns);
    c->data = c_data;
    return c;
}

//function to extract eigenvalues and eigenvectors from tensor
tensor* decompose_tensor(matrix* tensor_matrix, const double min_diffusitivity){
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

//Fits matrix using svd method
double* fitter(matrix* design, double* weights, double* signal, size_t sig_size){
    gsl_vector* signal_gsl = gsl_vector_alloc(sig_size);
    int i;
    for (i = 0; i < sig_size; i++){
        gsl_vector_set(signal_gsl, i, signal[i] * weights[i]);
    }
    matrix* weighted_design = matrix_scale(design, weights, 1);
    gsl_matrix* weighted_design_gsl = to_gsl(weighted_design);
    gsl_matrix* V = gsl_matrix_alloc(design->columns, design->columns);
    gsl_vector* S = gsl_vector_alloc(design->columns);
    gsl_vector* work = gsl_vector_alloc(design->columns);
    gsl_linalg_SV_decomp(weighted_design_gsl, V, S, work);
    gsl_vector* output_gsl = gsl_vector_alloc(design->columns);
    gsl_linalg_SV_solve(weighted_design_gsl, V, S, signal_gsl, output_gsl);
    double* output = malloc(sizeof(double) * design->columns);
    for(i=0;i<design->columns;i++){
        output[i] = gsl_vector_get(output_gsl, i);
    }
    gsl_vector_free(signal_gsl);
    gsl_matrix_free(weighted_design_gsl);
    gsl_matrix_free(V);
    gsl_vector_free(S);
    gsl_vector_free(work);
    gsl_vector_free(output_gsl);
    free_matrix(weighted_design);
    return output;
}
