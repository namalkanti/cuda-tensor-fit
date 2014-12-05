cdef extern from "cython_interface.h":

    void python_to_c(double* ols_fit_data, int ols_rows, int ols_columns, 
                    double* design_matrix_data, int design_rows, int design_columns,
                    double* signal_data, int signals, int signal_elements,
                    double min_signal, double min_diffusivity, double* output)
