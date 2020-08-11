def get_event_type(file_sess):
    """
    Argument: subject session header file => numpy.ndarray object (e.g. file['A01T_HDR'])
    Return: numpy array with shape (number of events, 1), dtype=uint16
    
    Note: 
    Each session (each file) = 6 runs 
    (Event = 32766 = Start new trial. i.e. will see 6x 32766 event type per file)
    Each run = 48 trials (i.e. will see 48x 768 event type between each 32766 event type)
    Each trial = 1 event (i.e. will see event 769, 770, 771 or 772 between each 768)

    Event Types: 
    768 : Start Trial
    769 : Left (Class 1)
    770 : Right (Class 2)
    771 : Foot (Class 3)
    772 : Tongue (Class 4)
    32766 : Start new trial (i.e. between runs)

    276 : EEG Idle: Eyes Open
    277 : EEG Idle: Eyes Closed
    783 : Cue unknown
    1023: Reject
    1072: Eyes Movement
    """
    # Check correctness of input
    assert file_sess[0,0][0][0] == 'GDF', 'Input format is wrong'  
    
    # Check length (=num events) to be at least 48x6x2 + 6 = 582 (6 runs. 48 trials per run. Each run has start trial + movt at least)
    assert len(file_sess['EVENT'][0][0][0][0][1]) >= 48*6*2 + 6 , 'Num of logged events too small for 48 trials in 6 sessions'  

    return file_sess['EVENT'][0][0][0][0][1]




def get_position(file_sess):
    """
    Argument: subject session header file => numpy.ndarray object (e.g. file['A01T_HDR'])
    Return: numpy array with shape (number of events, 1), dtype=uint16
    
    Sample Rate = 250Hz
    Number of Runs = 6
    Number of Trials / run = 48
    Duration of each trial ~ 7.5s
    """
    # Check correctness of input
    assert file_sess[0,0][0][0] == 'GDF', 'Input format is wrong' 
    
    position = file_sess['EVENT'][0][0][0][0][2]   # A0xX_HDR.EVENT.POS ->  Position of the Event 
    
    # Check length (=num events) to be at least 48x6x2 + 6 = 582 
    # (6 runs. 48 trials per run. Each run has start trial + movt at least)
    assert len(position) >= 48*6*2 + 6 , 'Num of logged events is too small for 48 trials in 6 sessions'  
    
    # Check duration of recording should be > 48 x 6 x 7.5s = 2160s, but less than 4000s (arbitrarily chosen)
    assert position[-1]/250 >= 48*6*7.5, 'Recording is too short'
    assert position[-1]/250 < 4000, 'Recording is too long'
    
    return position


def get_duration(file_sess):
    """
    Argument: subject session header file => numpy.ndarray object (e.g. file['A01T_HDR'])
    Return: numpy array with shape (number of events, 1), dtype=uint16
    
    Sample Rate = 250Hz
    Number of Runs = 6
    Number of Trials / run = 48
    Duration of each trial ~ 7.5s
    """
    # Check correctness of input
    assert file_sess[0,0][0][0] == 'GDF', 'Input format is wrong' 
    
    duration = file_sess['EVENT'][0][0][0][0][3]   # A0xX_HDR.EVENT.DUR  -> Duration of the Event

    # Check length (=num events) to be at least 48x6x2 + 6 = 582 
    # (6 runs. 48 trials per run. Each run has start trial + movt at least)
    assert len(duration) >= 48*6*2 + 6 , 'Num of logged events is too small for 48 trials in 6 sessions' 
    
    return duration


def make_x_y(sampled_eeg, eeg_events):
    """
    Argument:
        sampled_eeg: subject session header file => numpy array (e.g. file['A01T_s'])
        eeg_events: subject session header file => numpy.ndarray object (e.g. file['A01T_HDR'])
        
    Return:
        
    """
    import numpy as np
    
    # Check that input sampled_EEG has shape (samples, 25)
    assert sampled_eeg.shape[1] == 25, 'Input sampled_eeg is not of shape (samples, 25)'
    
     # Check correctness of header input
    assert eeg_events[0,0][0][0] == 'GDF', 'Input eeg_events has wrong format' 
    
    # Get position, duratoin and events from eeg_events    
    position = get_position(eeg_events)
    duration = get_duration(eeg_events)
    eeg_event = get_event_type(eeg_events)
    
    