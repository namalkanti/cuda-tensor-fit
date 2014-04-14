#include <Python.h>
#include <numpy/arrayobject.h>

#include "fit_tensor.h"

//Helper functions
matrix* copy_numpy_array_into_matrix(PyObject* array);
void load_tensors_into_output(tensor** tensors, PyObject* output_array);

//Docstrings
static char module_docstring[] = "Weight Least Squares tensor fitting for dipy."

static char fit_tensor_docstring[] = "Takes in ols_fit matrix, design_matrix, signal, min_signal, min_diffusitivity, and dti_params(results) Does a weighted least squares approximation and returns results in dti_params."

static PyObject* fit_tensors(PyObject *self, PyObject *args){

    double min_sig, min_diffusitivity;
    PyObject *ols_fit, *design_matrix, *sig, *dti_params;

    if (!PyArg_ParseTuple(args, "000dd0", &ols_fit, &design_matrix, &sig, min_sig, 
                min_diffusitivity, &dti_params)){
        return NULL;
    }

    matrix* ols_fit_matrix = ;
    matrix* design_matrix = ;
    matrix* signal_matrix = ;
    tensor** output_tensors;
    fit_tensor(ols_fit_matrix, design_matrix, signal_matrix, min_sig, min_diffusitivity, output_tensors);
    load_tensors_into_output(output_tensors, dti_params);
}

static PyMethodDef module_methods[] = {
    {"c_fit_tensors", fit_tensors, METH_VARARGS, fit_tensor_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_bloch(void){
    PyObject* m = PyInitModule("tensor_fitting", modules_methods, module_docstring);
    if (NULL == m){
        return;
    }
    import_array();
}

//Takes a numpy array and loads it into a matrix object.
matrix* copy_numpy_array_into_matrix(PyObject* array){
}

//Takes a tensors array and loads it into a numpy array.
void load_tensors_into_output(tensor** tensors, PyObject* output_array){
}
