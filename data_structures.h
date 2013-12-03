//Struct for matrices
//Matrices are always row major in C
//Vectors are also stored in row major, keep this in mind when converting back to Python
typedef struct{
    double* data;
    int rows;
    int columns;
}matrix;

//Struct for tensors
//Expected length of vals is 3
//Expected length of vecs is 3 with three element arrays
typedef struct {
    double* vals;
    matrix* vecs;
}tensor;

//Struct for padded arrays to be deployed for kernels.
typedef struct {
    double* values;
    int original_length;
    int current_length;
}padded_array;

//Struct for padding matrix.
typedef struct {
    matrix* matrix;
    int original_m;
    int original_n;
}padded_matrix;


