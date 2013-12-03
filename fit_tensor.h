#include "opt_util.h"

//Function that take in a complete signal matrix and fits it
void fit_complete_signal(matrix* ols_fit, matrix* design_matrix, matrix* signal, double min_signal, double min_diffusivity, tensor** tensor_output);
