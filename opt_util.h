#include "structure_util.h"

/*
 * Iterates through an array and if value is less than min signal,
 * replaces with minimum value. After replacement check, will 
 * take the logarithm of the resulting value.
 */
void cutoff_log(double* signal, double min_signal, int length);

/*
 * Passes in each value to exp function and replaces it
 * in the returned array.
 */
double* exp_array(double const* input, int lenght);

/*
 * Maps a max operation across matrix with min_diffusitivity, then
 * returns a tensor from a symmetric 3x3 matrix.
 */
tensor* decompose_tensor_matrix(matrix const* tensor_matrix, double min_diffusitivity);

/*
 * Weights and does a least squares approximation on the signal.
 */
double* fit_matrix(matrix const* design_matrix, double const* weights, double const* signal, int signal_length);
