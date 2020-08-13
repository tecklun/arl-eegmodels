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


def add_label_to_eeg(sampled_eeg, event_type, position, duration):
    """
    Function to add label to the sampled eeg data (sampled_eeg), using information from header file. 
    Function also checks for reject events and reject label accordingly. 
    
    Argument:
        sampled_eeg: subject session header file => numpy.ndarray (e.g. file['A01T_s']) with shape (samples, 25)
        event_type: output of get_event_type. numpy.ndarray array of shape (num_events, 1)
        position: output of get_position. numpy.ndarray array of shape (num_events, 1)
        duration: output of get_position. numpy.ndarray array of shape (num_events, 1)
        
    Return:
        output: numpy.ndarray with shape (samples, 26), where the added columnn [index=25] gives the label on the sampled 
                data. The labels are :
                    0 : No class
                    1 : Left
                    2 : Right
                    3 : Foot
                    4 : Tongue
                The output will need to pass through another process to break it into [0.5s, 2.5s] post cue onset 
                (i.e. take time from 2.5s to 5s from sampled_eeg) for each trial with valid label. Those with no valid label
                (i.e. label = 0, no class) will be rejected, not used for training.
                
        val_trial: numpy.ndarray with shape (samples, 1), where:
            1 means that sample is a valid sample in a run. 
            0 means non-valid sample, either due to:
                reject event (event = 1023), or
                non-trial sample (e.g. in between start of new run to new trial)
    """
    import numpy as np
    
    # Check using assert on correctness of input 
    assert_add_label_to_eeg(sampled_eeg, event_type, position, duration)
    
    # Initialize last column class label (last_col) and valid trial label (val_trial)
    last_col = np.zeros((sampled_eeg.shape[0],1))
    val_trial = np.zeros((sampled_eeg.shape[0],1))
    
    # Create last column class label and val_trial
    for i in range(len(event_type)):
        
        if event_type[i,0] == 769 :   # Class 1: Left
            last_col[position[i,0]:position[i,0]+duration[i,0]] = 1
            
        elif event_type[i,0] == 770 :   # Class 2: Right
            last_col[position[i,0]:position[i,0]+duration[i,0]] = 2
        
        elif event_type[i,0] == 771 :   # Class 3: Foot
            last_col[position[i,0]:position[i,0]+duration[i,0]] = 3
            
        elif event_type[i,0] == 772 :   # Class 4: Tongue
            last_col[position[i,0]:position[i,0]+duration[i,0]] = 4
            
        elif event_type[i,0] == 768 :   # Start trial
            val_trial[position[i,0]:position[i,0]+duration[i,0]] = 1
        
        elif event_type[i,0] == 1023 :   # Reject event
            val_trial[position[i,0]:position[i,0]+duration[i,0]] = 0  
            
            
            
    # Mask last_col where val_trial == 0
    last_col[val_trial == 0] = 0
        
    # Create output dataset
    output = np.hstack((sampled_eeg, last_col))

    return output, val_trial


def assert_add_label_to_eeg(sampled_eeg, event_type, position, duration):
    # Check that input sampled_EEG has shape (samples, 25)
    assert sampled_eeg.shape[1] == 25, 'Input sampled_eeg is not of shape (samples, 25)'
    
     # Check correctness of event_type, position and duration
    assert len(event_type.shape) == 2 and event_type.shape[1] == 1, 'event_type of wrong input format'
    assert len(position.shape) == 2 and position.shape[1] == 1, 'position of wrong input format'
    assert len(duration.shape) == 2 and duration.shape[1] == 1, 'duration of wrong input format'
    
    # Check event_type, position, duration have same shape
    assert (event_type.shape == duration.shape) and (duration.shape == position.shape), 'event_type, duration, position input have different shape'
    
    
    
def split_dataset(eeg_w_label, val_trial, start_pos, duration):
    """
    Function splits the eeg_w_label dataset to the respective training / test samples, 
    using val_trial as reference on the trial start / stop. start_pos and duration to determine trimming the dataset if provided.
    
    Argument:
        eeg_w_label: numpy.ndarray with shape (samples, 26), where last column is the label
        val_trial: numpy.ndarray with shape (samples, 1), denoting 1 for samples collected during valid trial, 0 otherwise
        start_pos: integer to denote offset from each val_trial to start breaking data up
        duration: integer to denote the length (duration) of each valid trial to use.
    """