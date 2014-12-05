import numpy as np
cimport numpy as np
from cython_interface cimport python_to_c

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

def tensor_fit(ols_fit, design_matrix, signal, float min_signal, float min_diffusivity):
    """
    Cython api for fit tensort. 
    Will be compiler against either CPU or GPU target.
    """

    ols_rows = ols_fit.shape[0]
    ols_columns = ols_fit.shape[1]
    design_rows = design_matrix.shape[0]
    design_columns = design_matrix.shape[1]
    signals = signal.shape[0]
    signal_elements = signal.shape[1]

    output = np.zeros((signals, 4, 3))

    python_to_c(<double*> np.PyArray_DATA(ols_fit), <int> ols_rows, <int> ols_columns,
                <double*>np.PyArray_DATA(design_matrix), <int> design_rows, <int> design_columns,
                <double*>np.PyArray_DATA(signal), <int> signals, <int> signal_elements,
                min_signal, min_diffusivity, <double*> np.PyArray_DATA(output))

    return output

