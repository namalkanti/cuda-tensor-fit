#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include "fit_tensor_util.h"

//emulates numpy's maximum function and log function combined for efficiency.
//iterates through array and if value is less than min signal, replaces with minimum value.
//also takes logarithm of every value.
void cutoff_log(double* signal, double min_signal, int n){
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
double* exp_array(double* input, int n){
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

//Fits matrix using transpose and inverse method
//Originally used svd method
double* fitter(matrix* design, double* weights, double* signal, int sig_size){
    gsl_vector* signal_gsl = gsl_vector_alloc(sig_size);
    double* signal_clone = array_clone(signal, sig_size);
    matrix signal_mat = {signal_clone, sig_size, 1};
    int i;
    for (i = 0; i < sig_size; i++){
        signal_mat.data[i] = signal[i] * weights[i]; 
        gsl_vector_set(signal_gsl, i, signal[i] * weights[i]);
    }
    matrix* weighted_design = matrix_scale(design, weights, 1);
    gsl_matrix* weighted_design_gsl = to_gsl(weighted_design);
    gsl_matrix* trans_gsl = gsl_matrix_alloc(weighted_design_gsl->size2, weighted_design_gsl->size1);	
    gsl_matrix_transpose_memcpy(trans_gsl, weighted_design_gsl);
    matrix* trans = to_matrix(trans_gsl);
    matrix* fit = matrix_dot(trans, weighted_design);
    matrix* inter_signal = matrix_dot(trans,&signal_mat);
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

bool arr_compare(double* arr1, double* arr2, int n, double err){
    int i;
    for(i = 0; i < n; i++){
        if(fabs(arr1[i] - arr2[i]) > err)
                return false;
    }
    return true;
}

bool float_array_compare(float* array1, float* array2, int array_length, float margin){
    int i;
    for(i = 0; i < array_length; i++){
        if(fabs(array1[i] - array2[i]) > margin)
                return false;
    }
    return true;
}	

double* array_combine(double* arr1, int len1, double* arr2, int len2){
    double* result = malloc(sizeof(double) * len1 * len2);
    int i;
    for(i = 0; i < len1; i++){
        result[i] = arr1[i];
    }
    for(i = len1; i < (len1 + len2); i++){
        result[i] = arr2[i - len1];
    }
    return result;
}

double* array_clone(double* arr1, int n){
    double* result = malloc(sizeof(double) * n);
    int i;
    for(i = 0; i < n; i++){
        result[i] = arr1[i];
    }
    return result;
}

bool mat_compare(matrix* mat1, matrix* mat2, double err){
    if (mat1->rows != mat2->rows || mat1->columns != mat2->columns){
        return false;
    }
    if (!arr_compare(mat1->data, mat2->data, mat1->rows * mat2->columns, err)){
        return false;
    }
    return true;
}

//Helper function to negate an entire array
//Same issues as previous, brute force constrained to size, refactor
void negate(double* arr){
    int i;
    for(i = 0;i < 3; i++){
        arr[i] = arr[i] * -1;
    }
}

//Special comparison function to compare columnar eigenvalues and eigenvectors
//Regular comparison won't work since you can multiply the entire vector by negative and still have be valid
//But most comparison functions will reject it
//Brute force method, only works for 3 by 3 matrices
//Quick and dirty fix, should refactor later
bool columnar_eig_compare(matrix* mat1, matrix* mat2, double err){
    double vec1a[] = {mat1->data[0], mat1->data[3], mat1->data[6]};
    double vec1an[] = {mat1->data[0], mat1->data[3], mat1->data[6]};
    negate(vec1an);
    double vec2a[] = {mat1->data[1], mat1->data[4], mat1->data[7]};
    double vec2an[] = {mat1->data[1], mat1->data[4], mat1->data[7]};
    negate(vec2an);
    double vec3a[] = {mat1->data[2], mat1->data[5], mat1->data[8]};
    double vec3an[] = {mat1->data[2], mat1->data[5], mat1->data[8]};
    negate(vec3an);
    double vec1b[] = {mat2->data[0], mat2->data[3], mat2->data[6]};
    double vec2b[] = {mat2->data[1], mat2->data[4], mat2->data[7]};
    double vec3b[] = {mat2->data[2], mat2->data[5], mat2->data[8]};
    if (!(arr_compare(vec1a, vec1b, 3, err) || arr_compare(vec1an, vec1b, 3, err))){
        return false;
    }
    if (!(arr_compare(vec2a, vec2b, 3, err) || arr_compare(vec2an, vec2b, 3, err))){
        return false;
    }
    if (!(arr_compare(vec3a, vec3b, 3, err) || arr_compare(vec3an, vec3b, 3, err))){
        return false;
    }
    return true;
}

bool compare_tensors(tensor* a, tensor* b, double err){
    if (!arr_compare(a->vals, b->vals, 3, err))
        return false;
    if (!columnar_eig_compare(a->vecs, b->vecs, err)){
        return false;
    }
    return true; 
}

//Function to free matrix data
void free_matrix(matrix* mat){
    free(mat->data);
    free(mat);
}

//Function to clone matrix
//Don't forget to free!
matrix* clone_matrix(matrix* mat){
    int elements = mat->rows * mat->columns;
    double* clone_data = malloc(sizeof(double) * elements);
    int i;
    for (i = 0; i < elements; i++){
        clone_data[i] = mat->data[i];
    }
    matrix* clone = malloc(sizeof(matrix));
    clone->data = clone_data;
    clone->rows= mat->rows;
    clone->columns = mat->columns;
    if (!mat_compare(mat, clone, MARGIN)){
        return NULL;
    }
    return clone;
}

//Function to free tensor
void free_tensor(tensor* tens){
    free(tens->vals);
    free_matrix(tens->vecs);
    free(tens);

}
//Function to pad array of floats to multiple and return
padded_float_array* pad_array(float* array, int array_length,  int multiple) {
    padded_float_array* padded_array = malloc(sizeof(padded_float_array));
    padded_array->original_length = array_length;
    int pad_length = multiple - (array_length % multiple);
    int new_length = array_length + pad_length;
    padded_array->current_length = new_length;
    float* padded_values = malloc(sizeof(float) * new_length);
    int i; 
    for(i = 0;i < array_length;i++){
        padded_values[i] = array[i];
    }
    for(i;i < new_length;i++){
        padded_values[i] = 0;
    }
    padded_array->values = padded_values;
    return padded_array;
}

//Function to remove array of floats from padded array
float* get_array_from_padded_array(padded_float_array* padded_array) {
    int original_length = padded_array->original_length;
    int i;
    float* extracted_array = malloc(sizeof(float) * original_length);
    for(i = 0; i < original_length; i++){
        extracted_array[i] = padded_array->values[i];
    }
    return extracted_array;
}

//Function to free memory from padded array
void free_padded_array(padded_float_array* pointer){
    free(pointer->values);
    free(pointer);
}

//Pads columns of matrices. Should be called after pad_rows
void static pad_columns(double* matrix_values, int old_value_length, int new_value_length) {
    int i = old_value_length;
    for(i;i < new_value_length; i++){
        matrix_values[i] = 0;
    }
}

//Pads rows of matrices. Should be called before pad_columns
void static pad_rows(double* old_matrix_values, double* new_matrix_values, int original_columns, 
        int new_columns, int original_rows) {
    int i;
    for(i = 0;i < original_rows; i++){
        int j;
        int index;
        for(j = 0; j < original_columns;j++){
            index = i * original_columns +j;
            new_matrix_values[index] = old_matrix_values[index];
        }
        for(j;j < new_columns;j++){
            index = i * original_columns +j;
            new_matrix_values[index] = 0;
        }
    }
}

//Function to pad matrix and return padded output, MATRIX MUST BE ROW ORDER
padded_matrix* pad_matrix(matrix* matrix_to_pad, int m_multiple, int n_multiple) {
    padded_matrix* padded_output = malloc(sizeof(padded_matrix));
    matrix* new_matrix = malloc(sizeof(matrix));
    int additional_rows = m_multiple - (matrix_to_pad->rows % m_multiple);
    int additional_columns = n_multiple - (matrix_to_pad->columns % n_multiple);
    int new_m = matrix_to_pad->rows + additional_rows;
    int new_n = matrix_to_pad->columns + additional_columns;
    double* padded_matrix_values = malloc(sizeof(double) * new_m * new_n);
    pad_rows(matrix_to_pad.data, padded_matrix_values, matrix_to_pad.rows, new_n, matrix_to_pad->rows);
    pad_columns(padded_matrix_values, matrix_to_pad->rows * matrix_to_pad->columns, new_m * new_n);
    new_matrix->data = padded_matrix_values;
    new_matrix->rows = new_m;
    new_matrix->columns = new_n;
    padded_output->matrix = new_matrix;
    padded_output->original_m = matrix_to_pad->rows;
    padded_output->original_n = matrix_to_pad->columns;
    return padded_output;
}

//Frees matrix pointer
void free_padded_matrix(padded_matrix* matrix_pointer){
    free_matrix(matrix_pointer->matrix);
    free(matrix_pointer);
}
