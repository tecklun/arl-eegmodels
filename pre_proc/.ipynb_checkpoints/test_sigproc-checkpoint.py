import sigproc
import unittest
import numpy as np

class Test_exponential_running_standardize(unittest.TestCase):
    """ Test class for function sigproc.exponential_running_standardize """
    """ Arguments: data, factor_new=0.001, init_block_size=None, eps=1e-4"""
    """
    1. test_no_init_block_sgl: single channel (i.e. data.shape=(sample, 1)), with init_block_size=None
    2. test_no_init_block_dbl: 2 channel (i.e. data.shape=(sample,2)), with init_block_size=None
    """
    
    # Add your test methods for sigproc.exponential_running_standardize here.
    def test_no_init_block_sgl(self):
        """
        1. test_no_init_block_sgl: single channel (i.e. data.shape=(sample, 1)), with init_block_size=None
        """
        import pandas as pd
        # Initialize
        data = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
        factor_new = 0.1
        init_block_size=None
        eps = 1e-6
        
        # Actual
        actual = sigproc.exponential_running_standardize(data, factor_new, init_block_size, eps)
        
        # Expected
        # Get mean
        mean = np.zeros((data.shape))
        mean[0,0] = data[0,0]
        for i in range(1, len(data)):
            mean[i,0] = factor_new*data[i,0] + (1-factor_new)*mean[i-1,0]
        
        diff = data - mean   # Find diff from mean
        squared = diff * diff
        
        # Get variance
        variance = np.zeros((data.shape))
        variance[0,0] = squared[0,0]
        for i in range(1, len(squared)):
            variance[i,0] = factor_new*squared[i,0] + (1-factor_new)*variance[i-1,0]
        
        # Get standardized
        standardized = diff / np.maximum(eps, np.sqrt(variance))
        
        expected = standardized
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)

        
    def test_no_init_block_dbl(self):
        """
        2. test_no_init_block_dbl: 2 channel (i.e. data.shape=(sample,2)), with init_block_size=None
        """
        import pandas as pd
        # Initialize
        data = np.array([[1,10],[2,10],[3,10],[4,40],[5,20],[6,10],[7,10],[8,10],[9,10],[10,10]])
        factor_new = 0.1
        init_block_size=None
        eps = 1e-6
        
        # Actual
        actual = sigproc.exponential_running_standardize(data, factor_new, init_block_size, eps)
        
        # Expected
        # Get mean
        mean = np.zeros((data.shape))
        mean[0,:] = data[0,:]
        for i in range(1, len(data)):
            mean[i,:] = factor_new*data[i,:] + (1-factor_new)*mean[i-1,:]
        
        diff = data - mean   # Find diff from mean
        squared = diff * diff
        
        # Get variance
        variance = np.zeros((data.shape))
        variance[0,:] = squared[0,:]
        for i in range(1, len(squared)):
            variance[i,:] = factor_new*squared[i,:] + (1-factor_new)*variance[i-1,:]
        
        # Get standardized
        standardized = diff / np.maximum(eps, np.sqrt(variance))
        
        expected = standardized
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        
        
        
        

class Test_bandpass_cnt(unittest.TestCase):
    """ Test class for function sigproc.bandpass_cnt """
    """ Arguments: data, low_cut_hz, high_cut_hz, fs, filt_order=3, axis=0, filtfilt=False"""
    """
    1. test_bandpass_sgl: Single Channel bandpass filter
    """
    
    # Add your test methods for sigproc.bandpass_cnt here.
    def test_bandpass_sgl(self):
        """
        1. test_bandpass_sgl: 
        """
        # Initialize
        data = np.array([[10],[10],[10],[10],[-10],[-10],[-10],[-10],[10],[10],[10],[10],[-10],[-10],[-10],[-10],[10],[10],[10],[10]])
        low_cut_hz = 10
        high_cut_hz = 20
        fs = 100
        
        # Actual
        signal = sigproc.bandpass_cnt(data, low_cut_hz, high_cut_hz, fs)
        actual = signal.shape
        print(signal)
        
        # Expected        
        expected = data.shape
        
        # Check correct shape
        np.testing.assert_array_equal(actual, expected)
        
    def test_bandpass_dbl(self):
        """
        2. test_bandpass_dbl: 
        """
        # Initialize
        data = np.array([[10, 10],[10, 10],[10, -10],[10, -10],[-10, 10],[-10, 10],[-10, -10],[-10, -10],\
                         [10, 10],[10, 10],[10, -10],[10, -10],[-10, 10],[-10, 10],[-10, -10],[-10, -10],[10, 10],[10, 10],[10, -10],[10, -10]])
        low_cut_hz = 10
        high_cut_hz = 30
        fs = 100
        
        # Actual
        signal = sigproc.bandpass_cnt(data, low_cut_hz, high_cut_hz, fs)
        actual = signal.shape
        print(signal)
        
        # Expected        
        expected = data.shape
        
        # Check correct shape
        np.testing.assert_array_equal(actual, expected)
        
if __name__ == '__main__':
    unittest.main(exit=False)
