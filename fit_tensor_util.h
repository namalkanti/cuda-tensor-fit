#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include "data.h"

//Emulates numpy's maximum function and log function combined for efficiency.
//Iterates through array and if value is less than min signal, replaces with minimum value.
//Also takes logarithm of every value.
void cutoff_log(double* signal, double min_signal, size_t n){
    int i;
    for (i = 0; i < n; i++){
        if (signal[i] < min_signal){
            signal[i] = min_signal;
        }
        signal[i] = log(signal[i]);
    }
}

//Raises every e to the power of every element in the input array and outputs new array
//Pointer returned is allocated on heap, free memory when done using it
double* exp_array(double* input, size_t n){
    double* output = (double*) malloc(sizeof(double) * n);
    int i;
    for(i = 0; i < n; i++){
        output[i] = pow(M_E, input[i]);
    }
    return output;
}

//Function takes input array and matches each index to a certain position in a 3 x 3 matrix
//Only uses first six elements of input array. Will fail if less than six are provided.
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

//Takes in a matrix and and an array
//Multiplies each matrix row element by it's corresponding array element and repeats for all rows if transpose flag is 0
//If transpose flag is 1 will multiply by each column instead
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

//Returns gsl matrix for interfacing with gsl library for BLAS and LAPACK
//Meant to be used a helper function
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

//Returns matrix from gsl matrix.
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

//Function to take two matrices, dot them, and return the result
matrix* matrix_dot(matrix* A, matrix* B){
    double* C_data = (double*) malloc(sizeof(double) * A->rows * B->columns); 
    matrix* C = (matrix*) malloc(sizeof(matrix));
    C->rows = A->rows;
    C->columns = B->columns;
    //cblas_dgemm();
    C->data = C_data;
    return C;
}
