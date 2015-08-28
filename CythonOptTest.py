import os.path
import unittest 

import numpy as np
import scipy as sp

from cython_opt import tensor_fit

#Test info
DATA = "test_data"
SIGNALS = "signal.npy"
OLS = "ols_fit.npy"
DESIGN = "design_matrix.npy"
MINS = "mins.npy"
EXPECTED = "tensors.npy" 

FILES = [SIGNALS, OLS, DESIGN, MINS, EXPECTED]

class CythonOptTest(unittest.TestCase):
    """
    Unit tests for tensor fit c extension.
    """

    def setUp(self):
        """
        Initializes arrays
        """
        self._signals, self._ols, self._design, self._mins, self._expected = [np.load(os.path.join(DATA, file_name)) for file_name in FILES]

    def test_tensor_fit(self):
        """
        Tests tensor fit function.
        """

        min_signal = self._mins[0]
        min_diffusivity = self._mins[1]
        result = tensor_fit(self._ols, self._design, self._signals, min_signal, min_diffusivity)
        expected_eigs = np.array([np.array(arr[0]) for arr in self._expected])
        result_eigs = np.array([np.array(arr[0]) for arr in result])
        self.assertTrue(np.allclose(expected_eigs, result_eigs))

if __name__ == "__main__":
    unittest.main()
