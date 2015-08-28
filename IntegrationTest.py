import subprocess
import unittest

import numpy as np


GIT_SWITCH = "~/cuda_dt/switch {0}"
MIN_SIGNAL = 1

import pdb

def switch_to_branch(branch_name):
    """
    Switches dipy library to git branch.
    """
    cmd = GIT_SWITCH.format(branch_name)
    subprocess.call(cmd, shell=True)


class IntegrationTest(unittest.TestCase):
    """
    Integration test from python interface for FA calculations.
    """

    def setUp(self):
        """
        Loads numpy arrays into matrices.
        """
        self._design = np.load("design_matrix.npy")
        self._data = np.load("data.npy")
        self._dti_params = np.load("dti_params.npy")

    def test_base(self):
        """
        Tests dti code with simple python.
        """
        switch_to_branch("master")
        from dipy_UCSF.reconst.dti import wls_fit_tensor
        results = wls_fit_tensor(self._design, self._data, MIN_SIGNAL) 
        result_eigs = np.array([np.array(arr[0:3]) for arr in results])
        expected_eigs = np.array([np.array(arr[0:3]) for arr in self._dti_params]) 
        are_equal = np.allclose(expected_eigs, result_eigs)
        self.assertTrue(are_equal)

    def test_c(self):
        """
        Tests dti code with C.
        """
        switch_to_branch("single")
        from dipy_UCSF.reconst.dti import wls_fit_tensor
        results = wls_fit_tensor(self._design, self._data, MIN_SIGNAL) 
        result_eigs = np.array([np.array(arr[0:3]) for arr in results])
        expected_eigs = np.array([np.array(arr[0:3]) for arr in self._dti_params]) 
        are_equal = np.allclose(expected_eigs, result_eigs)
        self.assertTrue(are_equal)

    def test_openmp(self):
        """
        Tests dti code with C and openmp.
        """
        switch_to_branch("openmp")
        from dipy_UCSF.reconst.dti import wls_fit_tensor
        results = wls_fit_tensor(self._design, self._data, MIN_SIGNAL) 
        result_eigs = np.array([np.array(arr[0:3]) for arr in results])
        expected_eigs = np.array([np.array(arr[0:3]) for arr in self._dti_params]) 
        are_equal = np.allclose(expected_eigs, result_eigs)
        self.assertTrue(are_equal)

    def test_cuda(self):
        """
        Tests dti code with C and CUDA
        """
        pass


if __name__ == "__main__":
    unittest.main()
