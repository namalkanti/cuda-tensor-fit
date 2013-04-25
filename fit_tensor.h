#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
