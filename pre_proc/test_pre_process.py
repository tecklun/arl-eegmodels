import pre_process
import unittest
import numpy as np

class Test_add_label_to_eeg(unittest.TestCase):
    """ Test class for function preprocess.add_label_to_eeg """
    """
    1. test_no_event: no event
    2. test_one_event: one event 769 (Class 1: Left). Check correct class, check correct duration
    3. test_two_events: two events 770 (Class 2: Right) and 772 (Class 4: Tongue). Check correct class, check correct duration
    4. test_same_event: event 771 (Class 3: Feet) appears twice. 
    5. test_reject_one: event 769 (Class 1: Left) with event 1023 (Reject). Check no class is tagged.
    6. test_reject_accept: event 770 (Class 2: Right) with event 1023 (Reject). Then twice event 770 (Class 2: Right).
    7. test_accept_reject: event 772 (Class 4: Tongue). Then event 771 (Class 3: Feet) w 1023 (Reject). Then event 769 (Class 1: Left).
    8. test_other_events: event 32766 (Start new trial). Then event 769 (Class 1: Left). 
    """
    
    # Add your test methods for preprocess.add_label_to_eeg here.
    def test_no_event(self):
        """
        1. test_no_event: no event
        """
        # Initialize
        sampled_eeg = np.random.randn(100,25)
        event_type = np.zeros((2,1))
        position = np.zeros((2,1))
        duration = np.zeros((2,1))
        
        # Actual
        actual, act_val = pre_process.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
        # Expected
        last_col = np.zeros((sampled_eeg.shape[0],1))
        expected = np.hstack((sampled_eeg, last_col))
        expected_val = np.zeros((sampled_eeg.shape[0],1))
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(act_val, expected_val)
    
    
    def test_one_event(self):
        """
         2. test_one_event: one event 769 (Class 1: Left). Check correct class, check correct duration
        """
        # Initialize
        sampled_eeg = np.random.randn(100,25)
        event_type = np.array([[768],[769]])    # 768: Start Trial, # 769: Left, Class 1
        position = np.array(  [[2],  [4]])           # Both starts at position 2 (i.e. at 8ms mark) 
        duration = np.array(  [[20], [9]])          # Duration of 768: 20; Duration of 769: 9
        
        # Actual
        actual, act_val = pre_process.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
        # Expected
        last_col = np.zeros((sampled_eeg.shape[0],1))
        last_col[4:4+9,0] = 1 
        expected = np.hstack((sampled_eeg, last_col))
        expected_val = np.zeros((sampled_eeg.shape[0],1))
        expected_val[2:2+20,0] = 1 
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(act_val, expected_val)
    
    def test_two_events(self):
        """
        3. test_two_events: two events 770 (Class 2: Right) and 772 (Class 4: Tongue). Check correct class, check correct duration
        """
        # Initialize
        sampled_eeg = np.random.randn(100,25)
        event_type = np.array([[768],[770],[768],[772]])    
        position = np.array(  [[5],  [10], [50], [60]])           
        duration = np.array(  [[30], [12], [40], [10]])          
        
        # Actual
        actual, act_val = pre_process.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
        # Expected
        last_col = np.zeros((sampled_eeg.shape[0],1))
        last_col[10:10+12,0] = 2
        last_col[60:60+10,0] = 4
        expected = np.hstack((sampled_eeg, last_col))
        expected_val = np.zeros((sampled_eeg.shape[0],1))
        expected_val[5:5+30,0] = 1 
        expected_val[50:50+40,0] = 1 
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(act_val, expected_val)
        
        
    def test_same_event(self):
        """
        4. test_same_event: event 771 (Class 3: Feet) appears twice. 
        """
        # Initialize
        sampled_eeg = np.random.randn(100,25)
        event_type = np.array([[768],[771],[768],[771]])    
        position = np.array([  [5],  [20], [50], [70]])           
        duration = np.array([  [30], [3],  [40], [10]])          
        
        # Actual
        actual, act_val = pre_process.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
        # Expected
        last_col = np.zeros((sampled_eeg.shape[0],1))
        last_col[20:20+3,0] = 3
        last_col[70:70+10,0] = 3
        expected = np.hstack((sampled_eeg, last_col))
        expected_val = np.zeros((sampled_eeg.shape[0],1))
        expected_val[5:5+30,0] = 1 
        expected_val[50:50+40,0] = 1 
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(act_val, expected_val)
        
    def test_reject_one(self):
        """
        5. test_reject_one: event 769 (Class 1: Left) with event 1023 (Reject). Check no class is tagged.
        """
        # Initialize
        sampled_eeg = np.random.randn(100,25)
        event_type = np.array([[768],[1023],[769]])    
        position = np.array([  [5],  [5],   [10]])           
        duration = np.array([  [30], [30],  [12]])          
        
        # Actual
        actual, act_val = pre_process.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
        # Expected
        last_col = np.zeros((sampled_eeg.shape[0],1))
        expected = np.hstack((sampled_eeg, last_col))
        expected_val = np.zeros((sampled_eeg.shape[0],1)) 
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(act_val, expected_val)
        
    def test_reject_accept(self):
        """
        6. test_reject_accept: event 770 (Class 2: Right) with event 1023 (Reject). Then twice event 770 (Class 2: Right).
        """
        # Initialize
        sampled_eeg = np.random.randn(100,25)
        event_type = np.array([[768],[1023],[770],[768],[770],[768],[770]])    
        position = np.array(  [[10], [10],  [12], [50], [52], [80], [85]])           
        duration = np.array(  [[30], [30],  [8],  [20], [5],  [15], [4]])          
        
        # Actual
        actual, act_val = pre_process.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
        # Expected
        last_col = np.zeros((sampled_eeg.shape[0],1))
        last_col[52:52+5,0] = 2
        last_col[85:85+4,0] = 2
        expected = np.hstack((sampled_eeg, last_col))
        expected_val = np.zeros((sampled_eeg.shape[0],1))
        expected_val[50:50+20,0] = 1 
        expected_val[80:80+15,0] = 1 
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(act_val, expected_val)
        
    def test_accept_reject(self):
        """
        7. test_accept_reject: event 772 (Class 4: Tongue). Then event 771 (Class 3: Feet) w 1023 (Reject). Then event 769 (Class 1: Left).
        """
        # Initialize
        sampled_eeg = np.random.randn(100,25)
        event_type = np.array([[768],[772],[768],[1023],[771],[768],[769]])    
        position = np.array(  [[10], [12], [50], [50],  [52], [80], [85]])           
        duration = np.array(  [[30], [8],  [20], [20],  [5],  [15], [4]])          
        
        # Actual
        actual, act_val = pre_process.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
        # Expected
        last_col = np.zeros((sampled_eeg.shape[0],1))
        last_col[12:12+8,0] = 4
        last_col[85:85+4,0] = 1
        expected = np.hstack((sampled_eeg, last_col))
        expected_val = np.zeros((sampled_eeg.shape[0],1))
        expected_val[10:10+30,0] = 1 
        expected_val[80:80+15,0] = 1 
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(act_val, expected_val)
        
    def test_other_events(self):
        """
        8. test_other_events: event 32766 (Start new trial). Then event 769 (Class 1: Left). 
        """
        # Initialize
        sampled_eeg = np.random.randn(100,25)
        event_type = np.array([[32766],[768],[769]])    
        position = np.array(  [[10],   [80], [85]])           
        duration = np.array(  [[20],   [15], [4]])          
        
        # Actual
        actual, act_val = pre_process.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
        # Expected
        last_col = np.zeros((sampled_eeg.shape[0],1))
        last_col[85:85+4,0] = 1
        expected = np.hstack((sampled_eeg, last_col))
        expected_val = np.zeros((sampled_eeg.shape[0],1))
        expected_val[80:80+15,0] = 1 
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(act_val, expected_val)
        

        
# remove_columns        
class Test_remove_columns(unittest.TestCase):
    """ Test class for function preprocess.remove_columns """
    """
    1. test_remove_last_of_2: of 2 colummns, remove the last column
    2. test_remove_first_of_2: of 2 columns, remove the first column
    3. test_remove_last2_of_4: of 4 columns, remove last 2 columns
    4. test_remove_123_of_5: of 5 columns, remove columns index 1,2,3
    5. test_remove_2nd_of_5: of 5 columns, remove column index 1
    """
    
    # Add your test methods for preprocess.remove_columns here.
    def test_remove_last_of_2(self):
        """
        1. test_remove_last_of_2: of 2 colummns, remove the last column
        """
        # Initialize
        eeg_w_label = np.array([[1 , 2],
                                [3 , 4],
                                [5 , 6],
                                [7 , 8]])
        start_col_to_remove = 1
        end_col_to_remove = 1
        
        # Actual
        actual = pre_process.remove_columns(eeg_w_label, start_col_to_remove, end_col_to_remove)
        
        # Expected
        expected = np.array([[1],
                             [3],
                             [5],
                             [7]])
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        
        
    def test_remove_first_of_2(self):
        """
        2. test_remove_first_of_2: of 2 columns, remove the first column
        """
        # Initialize
        eeg_w_label = np.array([[1 , 2],
                                [3 , 4],
                                [5 , 6],
                                [7 , 8]])
        start_col_to_remove = 0
        end_col_to_remove = 0
        
        # Actual
        actual = pre_process.remove_columns(eeg_w_label, start_col_to_remove, end_col_to_remove)
        
        # Expected
        expected = np.array([[2],
                             [4],
                             [6],
                             [8]])
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
    
    def test_remove_last2_of_4(self):
        """
        3. test_remove_last2_of_4: of 4 columns, remove last 2 columns
        """
        # Initialize
        eeg_w_label = np.array([[1 , 2 , 10, 20],
                                [3 , 4 , 30, 40],
                                [5 , 6 , 50, 60],
                                [7 , 8 , 70, 80]])
        start_col_to_remove = 2
        end_col_to_remove = 3
        
        # Actual
        actual = pre_process.remove_columns(eeg_w_label, start_col_to_remove, end_col_to_remove)
        
        # Expected
        expected = np.array([[1,2],
                             [3,4],
                             [5,6],
                             [7,8]])
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
    
    def test_remove_123_of_5(self):
        """
        4. test_remove_123_of_5: of 5 columns, remove columns index 1,2,3
        """
        # Initialize
        eeg_w_label = np.array([[1 , 2 , 10, 20, 90],
                                [3 , 4 , 30, 40, 100],
                                [5 , 6 , 50, 60, 110],
                                [7 , 8 , 70, 80, 120]])
        start_col_to_remove = 1
        end_col_to_remove = 3
        
        # Actual
        actual = pre_process.remove_columns(eeg_w_label, start_col_to_remove, end_col_to_remove)
        
        # Expected
        expected = np.array([[1, 90],
                             [3, 100],
                             [5, 110],
                             [7, 120]])
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)

    def test_remove_2nd_of_5(self):
        """
        5. test_remove_2nd_of_5: of 5 columns, remove column index 1
        """
        # Initialize
        eeg_w_label = np.array([[1 , 2 , 10, 20, 90],
                                [3 , 4 , 30, 40, 100],
                                [5 , 6 , 50, 60, 110],
                                [7 , 8 , 70, 80, 120]])
        start_col_to_remove = 1
        end_col_to_remove = 1
        
        # Actual
        actual = pre_process.remove_columns(eeg_w_label, start_col_to_remove, end_col_to_remove)
        
        # Expected
        expected = np.array([[1 , 10, 20, 90],
                             [3 , 30, 40, 100],
                             [5 , 50, 60, 110],
                             [7 , 70, 80, 120]])
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        
        
# to_microvolt        
class Test_to_microvolt(unittest.TestCase):
    """ Test class for function preprocess.to_microvolt """
    """
    1. test_mult_channel
    """
    
    # Add your test methods for preprocess.to_microvolt here.
    def test_mult_channel(self):
        """
        1. test_mult_channel:
        """
        # Initialize
        eeg_sample = np.array([ [1 , 2],
                                [3 , 4],
                                [5 , 6],
                                [7 , 8]])
        
        # Actual
        actual = pre_process.to_microvolt(eeg_sample)
        
        # Expected
        expected = np.array([[1e6, 2e6],
                             [3e6, 4e6],
                             [5e6, 6e6],
                             [7e6, 8e6]])
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        
# apply_pre_proc        
class Test_apply_pre_proc(unittest.TestCase):
    """ Test class for function preprocess.apply_pre_proc """
    """
    1. test_5ch_remove_2: from 5 channels, remove index 3,4. Do all preproc
    """
    
    # Add your test methods for preprocess.apply_pre_proc here.
    def test_5ch_remove_2(self):
        """
        1. test_5ch_remove_2: from 5 channels, remove index 3,4. Do all preproc
        """
        import numpy as np
        import sigproc
        # Initialize
        eeg_sample = np.array([ [1 , 2 , 10, 20, 90],
                                [3 , 4 , 30, 40, 100],
                                [5 , 6 , 50, 60, 110],
                                [7 , 8 , 70, 80, 120]])
        start_remove_index=3
        end_remove_index=4
        
        # Actual
        actual, actual_val = pre_process.apply_pre_proc(eeg_sample, 0, start_remove_index, end_remove_index)
        
        # Expected
        expected = np.array([ [1 , 2 , 10],
                              [3 , 4 , 30],
                              [5 , 6 , 50],
                              [7 , 8 , 70]])  # Drop index 3,4
        expected = expected*1e6       # Convert to uV
        expected = sigproc.bandpass_cnt(expected, 4, 38, 250)    # BPF 3-38Hz, Sampling Rate=250Hz
        
        factor_new = 1e-3
        eps = 10e-6
        
        # Get mean
        mean = np.zeros((expected.shape))
        mean[0,:] = expected[0,:]
        for i in range(1, len(expected)):
            mean[i,:] = factor_new*expected[i,:] + (1-factor_new)*mean[i-1,:]
        
        diff = expected - mean   # Find diff from mean
        squared = diff * diff
        
        # Get variance
        variance = np.zeros((expected.shape))
        variance[0,:] = squared[0,:]
        for i in range(1, len(squared)):
            variance[i,:] = factor_new*squared[i,:] + (1-factor_new)*variance[i-1,:]
        
        # Get standardized
        standardized = diff / np.maximum(eps, np.sqrt(variance))
        
        # Get expected output
        last_col = np.zeros((eeg_sample.shape[0], 1))
        expected = np.hstack((standardized, last_col))
        
        
        expected_val = np.zeros((eeg_sample.shape[0], 1))
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_val, expected_val)
        
        
        
if __name__ == '__main__':
    unittest.main(exit=False)