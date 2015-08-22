import numpy as np
cimport numpy as np
from cython_interface cimport python_to_c

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
GPU_CHUNKS = 6

def generate_indices(array_length, number_of_chunks):
    """
    Returns a list of indices that partition the array of length array_length
    into sub arrays of length size. If the array does not evenly divide then
    the last element will cover the rest of the elements. The second array 
    has sizes which will all be the same except the last one.

    >>> generate_indices(10, 3)
    ((0, 4), (4, 7), (7, 10)), (4, 3, 3) 
    >>> generate_indices(10, 4)
    ((0, 3), (3, 6), (6, 8), (8, 10)), (3, 3, 2, 2) 
    """
    numbers = np.range(array_length)
    splits = np.array_split(numbers, number_of_chunks)
    sizes = [len(x) for x in splits]
    indices = tuple([(split[0], split[-1]+1) for split in splits])
    return indices, sizes

def tensor_fit(ols_fit, design_matrix, signal, float min_signal, float min_diffusivity):
    """
    Cython api for fit tensort. 
    Will be compiler against either CPU or GPU target.
    """

    cdef int i

    ols_rows = ols_fit.shape[0]
    ols_columns = ols_fit.shape[1]
    design_rows = design_matrix.shape[0]
    design_columns = design_matrix.shape[1]
    signal = signal.astype(float)
    signals = signal.shape[0]
    signal_elements = signal.shape[1]

    output = np.zeros((signals, 4, 3))

    python_to_c(<double*> np.PyArray_DATA(ols_fit), <int> ols_rows, <int> ols_columns,
                <double*>np.PyArray_DATA(design_matrix), <int> design_rows, <int> design_columns,
                <double*>np.PyArray_DATA(signal), <int> signals, <int> signal_elements,
                min_signal, min_diffusivity, <double*> np.PyArray_DATA(output))

    return output

