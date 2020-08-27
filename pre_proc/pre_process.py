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
            last_col[position[i,0]:position-[i,0]+duration[i,0]] = 3
            
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
    # Check that input sampled_EEG has shape (samples, 22)
#     assert sampled_eeg.shape[1] == 22, 'Input sampled_eeg is not of shape (samples, 22)'
    
     # Check correctness of event_type, position and duration
    assert len(event_type.shape) == 2 and event_type.shape[1] == 1, 'event_type of wrong input format'
    assert len(position.shape) == 2 and position.shape[1] == 1, 'position of wrong input format'
    assert len(duration.shape) == 2 and duration.shape[1] == 1, 'duration of wrong input format'
    
    # Check event_type, position, duration have same shape
    assert (event_type.shape == duration.shape) and (duration.shape == position.shape), 'event_type, duration, position input have different shape'
    
    
    
def remove_columns(eeg_w_label, start_col_to_remove=22, end_col_to_remove=24):
    """
    Function to remove list of columns in col_to_remove from eeg_w_label. Returns eeg_w_label with removed columns.
    Argument: 
        eeg_w_label:   numpy.ndarray with shape (samples, 26), where last column [25] is the label. 
                       columns [22 to 24] are EOG channels which need to be removed
        start_col_to_remove: start index of columns to be removed. By default set to 22.
        end_col_to_remove: end index of columns to be removed. By default set to 24. 
    
    Returns: 
        eeg_w_label_cleaned: numpy.ndarray with shape (samples, 23), where last column [22] is the label.
                             It is the eeg_w_label dataset with removed columns
    """
    import numpy as np
    
    # Check correctness of input
    assert_remove_columns(eeg_w_label, start_col_to_remove, end_col_to_remove)
    
    # Start removing columns
    if start_col_to_remove == 0:    # If start removing from column 0
        eeg_w_label_cleaned = eeg_w_label[:,end_col_to_remove+1:].reshape(eeg_w_label.shape[0], eeg_w_label.shape[1]-end_col_to_remove-1)
    
    elif end_col_to_remove == eeg_w_label.shape[1]-1 : # If remove till ending columns
        eeg_w_label_cleaned = eeg_w_label[:,0:start_col_to_remove].reshape(eeg_w_label.shape[0], start_col_to_remove)
    
    else:     # If remove in between columns
        eeg_w_label_cleaned_front = eeg_w_label[:,0:start_col_to_remove].reshape(eeg_w_label.shape[0], start_col_to_remove)
        eeg_w_label_cleaned_back = eeg_w_label[:,end_col_to_remove+1:].reshape(eeg_w_label.shape[0], eeg_w_label.shape[1]-end_col_to_remove-1)

        eeg_w_label_cleaned = np.hstack((eeg_w_label_cleaned_front, eeg_w_label_cleaned_back)) 
    
    return eeg_w_label_cleaned


    
def assert_remove_columns(eeg_w_label, start_col_to_remove=22, end_col_to_remove=24):
    import numpy as np
    
    # Check start_col_to_remove and end_col_to_remove are integers, and that start is <= end
    assert (type(start_col_to_remove) == int) and (type(end_col_to_remove) == int), 'start/end_col_to_remove must be integers'
    assert start_col_to_remove <= end_col_to_remove, 'Start column must be smaller than end column'
    
    # Check eeg_w_label is of numpy.ndarray type and of shape (samples, columns)
    assert type(eeg_w_label) == np.ndarray, 'eeg_w_label must be numpy.ndarray'
    assert len(eeg_w_label.shape) == 2, 'eeg_w_label must be of shape (samples, columns)'
    
    # Check start_col_to_remove is positive and within range columns of eeg_w_label
    assert start_col_to_remove >= 0, 'start_col_to_remove must be non-negative'
    assert start_col_to_remove <= eeg_w_label.shape[1]-1, 'start_col_to_remove cannot be bigger than number of columns of eeg_w_label'
    
    # Check end_col_to_remove is positive and within range columns of eeg_w_label
    assert end_col_to_remove >= 0, 'end_col_to_remove must be non-negative'
    assert end_col_to_remove <= eeg_w_label.shape[1]-1, 'end_col_to_remove cannot be bigger than number of columns of eeg_w_label'
    

def to_microvolt(eeg_sample):
    """
    Function: Convert data from volts to microvolts
    Argument: eeg_sample - numpy.ndarray with shape (samples, channels), where channels = 22. In volts.
    Returns: eeg_sample_uV - numpy.ndarray with shape (samples, channels), where channels = 22. In microvolt
    """
    return eeg_sample*1e6
    
    
def apply_pre_proc(eeg_sample, eeg_header=0, start_remove_index=22, end_remove_index=24):
    """
    Function: Apply list of pre-processing techniques to eeg_sample, and create labels using eeg_header:
    On eeg_sample
    1. drop EOG channels: use remove_columns on eeg_samples to remove columns 22 to 24. Retaining columns 0 to 21 (i.e. 22 EEG electrodes)
    2. convert to uV: use to_microvolt on eeg_samples to convert from volts to microvolts
    3. bandpass filter: use sigproc.bandpass_cnt to perform BPF on signal
    4. exponential weighted average (EMA): use sigproc.exponential_running_standardize to perform EMA on signal
    
    On eeg_sample and eeg_header
    1. get event type: apply get_event_type on eeg_header to get event
    2. get position: apply get_position on eeg_header to get position
    3. get duration: apply get_duration on eeg_header to get duration
    4. add label to eeg_sample: apply add_label_to_eeg with event type, position and duration to generate label for eeg_sample
    
    Argument:
        eeg_sample: subject session header file => numpy.ndarray (e.g. file['A01T_s']) with shape (samples, 25)
        eeg_header: subject session header file => numpy.ndarray object (e.g. file['A01T_HDR'])
        start_remove_index: index of first EOG channel that needs to be removed. Default=22
        end_remove_index: index of last EOG channel that needs to be removed. Default=24
        
    Return:
        processed_eeg: eeg signal after pre-processing and label
        val_trial: numpy.ndarray with shape (samples, 1), where:
            1 means that sample is a valid sample in a run. 
            0 means non-valid sample, either due to:
                reject event (event = 1023), or
                non-trial sample (e.g. in between start of new run to new trial)
    """
    import numpy as np
    from pre_proc import sigproc  
    
    eeg_sample = remove_columns(eeg_sample, start_remove_index, end_remove_index)  # Remove EOG channels
    eeg_sample = to_microvolt(eeg_sample)  # Convert from volt to microvolt
#     eeg_sample = sigproc.bandpass_cnt(eeg_sample, 4., 38., 250.)   # BPF from 4Hz to 38Hz, EEG Sampling Rate: 250Hz
    eeg_sample = sigproc.exponential_running_standardize(eeg_sample, factor_new=1e-3, init_block_size=None, eps=1e-6)   # Apply EMA on signal
    
    if eeg_header is 0:   # No eeg_header. For testing of eeg_sample pre-processing
        last_col = np.zeros((eeg_sample.shape[0], 1))
        processed_eeg_w_label = np.hstack((eeg_sample, last_col))
        val_trial = np.zeros((eeg_sample.shape[0],1))
        
    else:    # The main program to add label
        event_type = get_event_type(eeg_header)
        position = get_position(eeg_header)
        duration = get_duration(eeg_header)
        processed_eeg_w_label, val_trial = add_label_to_eeg(eeg_sample, event_type, position, duration)
        
    return processed_eeg_w_label, val_trial





def split_dataset(processed_eeg_w_label, val_trial, start_pos=int((2)*250), duration=int(2*250), sgl_label=True):
    """
    Function splits the processed_eeg_w_label dataset to retain only the trial samples (i.e. val_trial == 1), 
    using val_trial as reference on the trial start / stop. start_pos and duration to determine trimming the dataset if provided.
    
    Argument:
        processed_eeg_w_label: numpy.ndarray with shape (samples, 23), where last column is the label
        val_trial: numpy.ndarray with shape (samples, 1), denoting 1 for samples collected during valid trial, 0 otherwise
        start_pos: integer to denote offset from each val_trial to start breaking data up. Default to 2.5x250 = 625. If 0, ignore this field
        duration: integer to denote the length (duration) of each valid trial to use. If 0, ignore this field
            Cue onset at 2s from trial start
            Epoch data at 0.5s to 2.5s from cue onset
            Thus, start_pos at 2.5s x 250 (Hz), duration at 2s x 250Hz
        sgl_label: boolean to decide on format of output Y. 
            If True (default): output Y with shape (trials, 1) for classification
            If False: output Y with shape (trials, samples_trial, 1) for sequence training
            
    Return:
        X: numpy.ndarray with shape (trials, samples_trial, channels=22), where all channels are EEG data. No label
        Y: numpy.ndarray with shape (trials, 1) if sgl_label=True, else shape (trials, samples_trial, 1). Output label 
    """
    import numpy as np

    # Check correctness of input
    assert_split_dataset(processed_eeg_w_label, val_trial, start_pos, duration)
    
    # Find location and split location from valid trial
    val_trial = val_trial.astype(np.int64)   # Convert all val_trial from float to integer
    
    ones = np.where(val_trial[:,0]==1)
    index_ones = np.array(ones)[0]
    index_ones_dly = np.array(ones)[:,1:][0]   # Get 1 delayed version of index_ones
    index_ones = index_ones[:-1]               # Trim index_ones to be same length as index_one_dly
    indexes_1 = index_ones_dly-index_ones      # indexes_1 if =
                                               #    1 -> Not reach end of valid sample
                                               #  > 1 -> Reached end of last valid sample. Is now a new valid sample, thus split
    split_list = np.array(np.where(indexes_1 != 1))[0] # Provides list of location to split the dataset 
    
    
    splited_dataset = np.array(np.split(processed_eeg_w_label[(val_trial==1)[:,0]], split_list+1))   # raw splited_dataset
    
    # Get data
    X, Y = [], [] 
    if len(splited_dataset.flatten()) >= 1:  # If exist valid splited_dataset
        for i in range(len(splited_dataset)):
            assert splited_dataset[i].shape[0] >= start_pos+duration, 'start_pos + duration is longer then length of split dataset'

            # X Value
            if start_pos == 0 and duration == 0:
                X.append(splited_dataset[i][:,:-1])
            else: 
                X.append(splited_dataset[i][start_pos:start_pos+duration,:-1])

            # Y Value
            if start_pos == 0 and duration == 0:
                y_labels = splited_dataset[i][:,-1]
            else: 
                y_labels = splited_dataset[i][start_pos:start_pos+duration,-1]
                
            if sgl_label:  # Single Label
                y_value = np.array([np.apply_along_axis(np.bincount, 0, y_labels.astype(np.int64)).argmax()]) 
                
                Y.append(y_value)
            else:          # Label per sample
                y_value = np.array(y_labels).astype(np.int64)
                Y.append(np.expand_dims(y_value, axis=1))
   
    # Output
    return np.array(X), np.array(Y)


def assert_split_dataset(processed_eeg_w_label, val_trial, start_pos=(2+0.5)*250, duration=2*250):
    
    import numpy as np
    
    # Check dtype and dimensions of input
    assert type(processed_eeg_w_label) == np.ndarray, 'processed_eeg_w_label must be of type np.ndarray'
    assert type(val_trial) == np.ndarray, 'val_trial must be of type np.ndarray'
    
    assert len(processed_eeg_w_label.shape) == 2, 'processed_eeg_w_label must be of shape (samples, channel+label)'
    assert (len(val_trial.shape) == 2 ) and (val_trial.shape[1] == 1), 'val_trial must be of shape (samples, 1)'
    assert(processed_eeg_w_label.shape[0] == val_trial.shape[0]), '.shape[0] of processed_eeg_w_label and val_trial must be equal'
    
    assert (type(start_pos) == int) and (type(duration) == int), 'start_pos and duration must be of type int'
    
    # Check value of start_pos + duration < samples
    assert (start_pos + duration) < val_trial.shape[0], 'start_pos + duration exceeds number of samples in input'
    
    # Check val_trial has only 0s and 1s and end with zeros
    assert (min(val_trial) >= 0) and (max(val_trial) <= 1), 'val_trial must only contain 0s and 1s'
    assert (val_trial[-1,0] == 0), 'val_trial must end with zeros'