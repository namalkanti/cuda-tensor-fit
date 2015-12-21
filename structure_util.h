#include <stdbool.h>
#include <gsl/gsl_matrix.h>
#include "data_structures.h"

//Array functions.

/*
 * Compares array a and b, returning true if corresponding values are within
 * the err of each other.
 */
bool array_compare(double const* a, double const* b, int length, double err);

/*
 * Fuction to combine two arrays, the first will be in front.
 */
double* array_combine(double const* a, double const* b, int alength, int blength); 

/*
 * Function clones an array and returns it.
 */
double* array_clone(double const* arr, int length);

/*
 * Rounds number up to multiple of 2.
 */
int round_to_power_of_two(int number);

//Matrix operations

/*
 * Function will take first six elements in array(fail otherwise)
 * and return a 3x3 symmetric matrix with this positioning.
 * 0 1 3
 * 1 2 4
 * 3 4 5
 */
matrix* get_lower_triangular(double const* input);

/*
 * Takes a the vector and multiples each row by each respective 
 * element in the vector(lenght of the vector has to equal N). If
 * trans is 1, do the same operation on the columns instead.
 */
matrix* scale_matrix(matrix const* mat, double const* vector, int trans);

/*
 * Returns the dot product of matrix a and matrix b
 */
matrix* matrix_dot(matrix const* a, matrix const* b);

/*
 * Returns a transposed matrix
 */
matrix* transpose(matrix const* mat);

/*
 * Compares values in matrix a and b, and returns true if they
 * differ within the given error.
 */
bool matrix_compare(matrix const* a, matrix const* b, double err);

/*
 * Clones matrix.
 */
matrix* clone_matrix(matrix const* mat);

/*
 * Creates an matrix on the heap and returns a pointer to it.
 */
matrix* create_matrix(double* data, int rows, int columns);

/*
 *Compares absolute value of columns  in 3x3 matrix with each other; 
 *if inside margin, return true.
 */
bool compare_eigenvalues_by_column(matrix const* a, matrix const* b, double err);

/*
 * Frees matrix structure.
 */
void free_matrix(matrix* mat);

//Gsl matrix operations

/*
 *Returns gsl matrix from matrix format.
 */
gsl_matrix* to_gsl(matrix const* mat);

/*
 * Returns a matrix from a gsl matrix.
 */
matrix* to_matrix(gsl_matrix const* gsl_mat);

//Tensor operations

/*
 * Compares every value in corresponding a and b and returns true if 
 * all values are within error value.
 */
bool compare_tensors(tensor const* a, tensor const* b, double err);

/*
 * Frees tensor structure a.
 */
void free_tensor(tensor* tens);
