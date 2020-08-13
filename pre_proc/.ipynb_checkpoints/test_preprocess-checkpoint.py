import preprocess
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
        actual, act_val = preprocess.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
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
        actual, act_val = preprocess.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
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
        actual, act_val = preprocess.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
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
        actual, act_val = preprocess.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
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
        actual, act_val = preprocess.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
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
        actual, act_val = preprocess.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
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
        actual, act_val = preprocess.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
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
        actual, act_val = preprocess.add_label_to_eeg(sampled_eeg, event_type, position, duration)
        
        # Expected
        last_col = np.zeros((sampled_eeg.shape[0],1))
        last_col[85:85+4,0] = 1
        expected = np.hstack((sampled_eeg, last_col))
        expected_val = np.zeros((sampled_eeg.shape[0],1))
        expected_val[80:80+15,0] = 1 
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(act_val, expected_val)
        
        
if __name__ == '__main__':
    unittest.main(exit=False)
