#include <math.h>
#include <gsl/gsl_cblas.h>
#include "structure_util.h"

//Declarations for helper functions
void static negate(double* arr);
void static pad_rows(double* old_matrix_values, double* new_matrix_values, int original_columns, 
        int new_columns, int original_rows); 
void static pad_columns(double* matrix_values, int old_value_length, int new_value_length); 

//Array function definitions
bool array_compare(double const* a, double const* b, int length, double err){
    int i;
    for(i = 0; i < length; i++){
        if(fabs(a[i] - b[i]) > err)
                return false;
    }
    return true;
}

double* array_combine(double const* a, double const* b, int alength, int blength){
    double* result = malloc(sizeof(double) * (alength + blength));
    int i;
    for(i = 0; i < alength; i++){
        result[i] = a[i];
    }
    for(i = alength; i < (alength + blength); i++){
        result[i] = b[i - alength];
    }
    return result;
}

double* array_clone(double const* arr, int length){
    double* result = malloc(sizeof(double) * length);
    int i;
    for(i = 0; i < length; i++){
        result[i] = arr[i];
    }
    return result;
}

int round_to_power_of_two(int number) {
    int estimate = 2;
    while ( estimate / number < 1){
        estimate *= 2;
    }
    return estimate;
}

//Definitions of matrix functions

matrix* get_lower_triangular(double const* input){
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

matrix* scale_matrix(matrix const* mat, double const* vector, int trans){
    double* output = (double*) malloc(sizeof(double) * (mat->rows * mat->columns));
    int i, j;
    int columns = mat->columns;
    switch(trans)
    {
        case 0:
            for(i = 0; i < mat->rows; i++){
                for(j = 0; j < columns; j++){
                    output[ i * columns + j] = mat->data[i * columns + j] * vector[j];
                }
            }
            break;
        case 1:
            for(i = 0; i < mat->rows; i++){
                for(j = 0; j < columns; j++){
                    output[i * columns + j] = vector[i] * mat->data[i * columns + j];
                }
            }
            break;
    }
    matrix* output_mat = (matrix*) malloc(sizeof(matrix));
    output_mat->data = output;
    output_mat->rows = mat->rows;
    output_mat->columns = mat->columns;
    return output_mat;
}

matrix* matrix_dot(matrix const* a, matrix const* b){
    double* c_data = malloc(sizeof(double) * a->rows * b->columns); 
    matrix* c = malloc(sizeof(matrix));
    c->rows = a->rows;
    c->columns = b->columns;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a->rows, b->columns, a->columns, 1.0, a->data, a->columns, b->data, b->columns, 0.0, c_data, c->columns);
    c->data = c_data;
    return c;
}

matrix* transpose(matrix const* mat){
    gsl_matrix* gsl_mat = to_gsl(mat);
    int result = gsl_matrix_transpose(gsl_mat);
    matrix* transposed = to_matrix(gsl_mat);
    gsl_matrix_free(gsl_mat);
    return transposed;
}

bool matrix_compare(matrix const* a, matrix const* b, double err){
    if (a->rows != b->rows || a->columns != b->columns){
        return false;
    }
    if (!array_compare(a->data, b->data, a->rows * b->columns, err)){
        return false;
    }
    return true;
}

matrix* clone_matrix(matrix const* mat){
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
    return clone;
}

matrix* create_matrix(double* data, int rows, int columns){
    matrix* pointer = malloc(sizeof(matrix));
    pointer->data = data;
    pointer->rows = rows;
    pointer->columns = columns;
    return pointer;
}

bool compare_eigenvalues_by_column(matrix const* a, matrix const* b, double err){
    double vec1a[] = {a->data[0], a->data[3], a->data[6]};
    double vec1an[] = {a->data[0], a->data[3], a->data[6]};
    negate(vec1an);
    double vec2a[] = {a->data[1], a->data[4], a->data[7]};
    double vec2an[] = {a->data[1], a->data[4], a->data[7]};
    negate(vec2an);
    double vec3a[] = {a->data[2], a->data[5], a->data[8]};
    double vec3an[] = {a->data[2], a->data[5], a->data[8]};
    negate(vec3an);
    double vec1b[] = {b->data[0], b->data[3], b->data[6]};
    double vec2b[] = {b->data[1], b->data[4], b->data[7]};
    double vec3b[] = {b->data[2], b->data[5], b->data[8]};
    if (!(array_compare(vec1a, vec1b, 3, err) || array_compare(vec1an, vec1b, 3, err))){
        return false;
    }
    if (!(array_compare(vec2a, vec2b, 3, err) || array_compare(vec2an, vec2b, 3, err))){
        return false;
    }
    if (!(array_compare(vec3a, vec3b, 3, err) || array_compare(vec3an, vec3b, 3, err))){
        return false;
    }
    return true;
}

void free_matrix(matrix* mat){
    free(mat->data);
    free(mat);
}

//Definitions of gsl functions

gsl_matrix* to_gsl(matrix const* mat){
    gsl_matrix* output = gsl_matrix_alloc(mat->rows, mat->columns);
    int i, j;
    for(i = 0; i < mat->rows; i++){
        for( j = 0; j < mat->columns; j++){
            gsl_matrix_set(output, i, j, mat->data[i * mat->columns + j]);
        }
    }
    return output;
}

matrix* to_matrix(gsl_matrix const* gsl_mat){
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

//Definitions of tensor functions

bool compare_tensors(tensor const* a, tensor const* b, double err){
    if (!array_compare(a->vals, b->vals, 3, err))
        return false;
    if (!compare_eigenvalues_by_column(a->vecs, b->vecs, err)){
        return false;
    }
    return true; 
}

void free_tensor(tensor* tens){
    free(tens->vals);
    free_matrix(tens->vecs);
    free(tens);

}

//Defintions of padded array functions.

padded_array* pad_array(double const* arr, int length,  int multiple) {
    padded_array* padded_arr = malloc(sizeof(padded_array));
    padded_arr->original_length = length;
    int pad_length = multiple - (length % multiple);
    int new_length = length + pad_length;
    padded_arr->current_length = new_length;
    double* padded_values = malloc(sizeof(double) * new_length);
    int i; 
    for(i = 0;i < length;i++){
        padded_values[i] = arr[i];
    }
    for(i;i < new_length;i++){
        padded_values[i] = 0;
    }
    padded_arr->values = padded_values;
    return padded_arr;
}

double* get_array_from_padded_array(padded_array const* padded_arr) {
    int original_length = padded_arr->original_length;
    int i;
    double* extracted_array = malloc(sizeof(double) * original_length);
    for(i = 0; i < original_length; i++){
        extracted_array[i] = padded_arr->values[i];
    }
    return extracted_array;
}

void free_padded_array(padded_array* pointer){
    free(pointer->values);
    free(pointer);
}

//Definitions of padded matrix functions

padded_matrix* pad_matrix(matrix const* matrix_to_pad, int m_multiple, int n_multiple) {
    padded_matrix* padded_output = malloc(sizeof(padded_matrix));
    matrix* new_matrix = malloc(sizeof(matrix));
    int additional_rows = m_multiple - (matrix_to_pad->rows % m_multiple);
    int additional_columns = n_multiple - (matrix_to_pad->columns % n_multiple);
    int new_m = matrix_to_pad->rows + additional_rows;
    int new_n = matrix_to_pad->columns + additional_columns;
    double* padded_matrix_values = malloc(sizeof(double) * new_m * new_n);
    pad_rows(matrix_to_pad->data, padded_matrix_values, matrix_to_pad->columns, new_n, matrix_to_pad->rows);
    pad_columns(padded_matrix_values, matrix_to_pad->rows * matrix_to_pad->columns, new_m * new_n);
    new_matrix->data = padded_matrix_values;
    new_matrix->rows = new_m;
    new_matrix->columns = new_n;
    padded_output->matrix = new_matrix;
    padded_output->original_m = matrix_to_pad->rows;
    padded_output->original_n = matrix_to_pad->columns;
    return padded_output;
}

void free_padded_matrix(padded_matrix* matrix_pointer){
    free_matrix(matrix_pointer->matrix);
    free(matrix_pointer);
}

/*
 * Helper function to negate an array.
 */
void static negate(double* arr){
    int i;
    for(i = 0;i < 3; i++){
        arr[i] = arr[i] * -1;
    }
}

/*
 * Function called to pad rows a row major matrix.
 * Should be called before pad_columns.
 */
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

/*
 * Function called to pad columns of a row major matrix.
 * Should be called after pad_rows.
 */
void static pad_columns(double* matrix_values, int old_value_length, int new_value_length) {
    int i = old_value_length;
    for(i;i < new_value_length; i++){
        matrix_values[i] = 0;
    }
}


