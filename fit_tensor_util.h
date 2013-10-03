#include <stdlib.h>
#include <stdbool.h>
#include <gsl/gsl_matrix.h>
#include "data_structures.h"
#define MARGIN .0000001

//emulates numpy's maximum function and log function combined for efficiency.
//iterates through array and if value is less than min signal, replaces with minimum value.
//also takes logarithm of every value.
void cutoff_log(double* signal, double min_signal, int n);

//raises every e to the power of every element in the input array and outputs new array
//pointer returned is allocated on heap, free memory when done using it
double* exp_array(double* input, int n);

//function takes input array and matches each index to a certain position in a 3 x 3 matrix
//only uses first six elements of input array. will fail if less than six are provided.
matrix* tensor_lower_triangular(double* input);

//takes in a matrix and and an array
//multiplies each matrix row element by it's corresponding array element and repeats for all rows if transpose flag is 0
//if transpose flag is 1 will multiply by each column instead
matrix* matrix_scale(matrix* input_matrix, double* vec, int trans);

//returns gsl matrix for interfacing with gsl library for blas and lapack
//meant to be used a helper function
gsl_matrix* to_gsl(matrix* mat);

//returns matrix from gsl matrix.
matrix* to_matrix(gsl_matrix* gsl_mat);

//function to take two matrices, dot them, and return the result
matrix* matrix_dot(matrix* a, matrix* b);

//function to extract eigenvalues and eigenvectors from tensor
tensor* decompose_tensor(matrix* tensor_matrix, const double min_diffusitivity);

//Fits matrix using svd method
double* fitter(matrix* design, double* weights, double* signal, int sig_size);

//Function to compare two arrays
bool arr_compare(double* arr1, double* arr2, int n, double err);

//Function to compare two floating point arrays
bool float_array_compare(float* array1, float* array2, int array_length, float margin);

//Function to combine two arrays
double* array_combine(double* arr1, int len1, double* arr2, int len2);

//Function to clone an array
double* array_clone(double* arr1, int n);

//Function to compare two matrices
bool mat_compare(matrix* mat1, matrix* mat2, double err);

//Helper function to negate an entire array
//Same issues as previous, brute force constrained to size, refactor
void negate(double* arr);

//Special comparison function to compare columnar eigenvalues and eigenvectors
//Regular comparison won't work since you can multiply the entire vector by negative and still have be valid
//But most comparison functions will reject it
//Brute force method, only works for 3 by 3 matrices
//Quick and dirty fix, should refactor later
bool columnar_eig_compare(matrix* mat1, matrix* mat2, double err);

//Function to compare two tensors
bool compare_tensors(tensor* a, tensor* b, double err);

//Function to free matrix data
void free_matrix(matrix* mat);

//Function to clone matrix
//Don't forget to free!
matrix* clone_matrix(matrix* mat);

//Function to free tensor
void free_tensor(tensor* tens);

//Function to pad array of floats to multiple and return
padded_float_array* pad_array(float* array, int array_length,  int multiple); 

//Function to remove array of doubles from padded array
double* get_array_from_padded_array(padded_double_array* padded_array);

//Function to free memory from padded array
void free_padded_array(padded_double_array* pointer);

//Function to pad matrix and return padded output
padded_matrix* pad_matrix(matrix* matrix_to_pad, int m_multiple, int n_multiple);

//Frees memory after use.
void free_padded_matrix(padded_matrix* matrix_pointer);

