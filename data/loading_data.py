import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import os, pickle
from data.translation import translation_dict

def get_experiment(experiment_number=10, data_dir=r"data"):
    if experiment_number not in range(10, 16):
        print(f"Experiment number {experiment_number} not valid. Must be 10-15.")
        return None
    
    file_name = f"sub-01_task-imagine_run-0{experiment_number:02d}_eeg.pkl"
    file_path = os.path.join(data_dir, file_name)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)  # data = list of 130 trial dictionaries
        print(f"Loaded {len(data)} trials from experiment {experiment_number}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def get_multiple_experiments(experiment_numbers=[10, 11, 12, 13, 14], data_dir=r"data"):
    all_data = {}  # all_data = {exp_num: list_of_trials, exp_num: list_of_trials, ...}
    
    # experiment_numbers = list of integers [10, 11, 12, 13, 14]
    for exp_num in experiment_numbers:  # exp_num = single integer like 10, 11, etc
        data = get_experiment(exp_num, data_dir)  # data = list of 130 trial dictionaries
        if data is not None:
            all_data[exp_num] = data  # Store list of trials under experiment number key
    return all_data

def extract_all_chinese_texts(experiment_numbers=[10, 11, 12, 13, 14], data_dir=r"data"):
    all_texts = set()  # all_texts = set of unique Chinese text strings
    
    # experiment_numbers = list of integers [10, 11, 12, 13, 14]
    for exp_num in experiment_numbers:  # exp_num = single integer like 10, 11, etc
        data = get_experiment(exp_num, data_dir)  # data = list of 130 trial dictionaries
        if data is not None:
            # data = list of trial dictionaries like [{'text': '...', 'input_features': array}, ...]
            for trial in data:  # trial = single dictionary with keys 'text' and 'input_features'
                all_texts.add(trial['text'].strip())  # Add Chinese text string to set
    
    return sorted(list(all_texts))  # Convert set to sorted list of Chinese text strings

def sample_trials_from_experiments(train_per_exp=15, test_per_exp=5, 
                                 experiment_numbers=[10, 11, 12, 13, 14], 
                                 data_dir=r"data"):
    train_trials = []  # train_trials = list of trial dictionaries for training
    test_trials = []   # test_trials = list of trial dictionaries for testing
    
    # experiment_numbers = list of integers [10, 11, 12, 13, 14]
    for exp_num in experiment_numbers:  # exp_num = single integer like 10, 11, etc
        data = get_experiment(exp_num, data_dir)  # data = list of 130 trial dictionaries
        if data is not None:
            # data = list of trial dictionaries, we slice to get specific trials
            train_trials.extend(data[:train_per_exp])  # Add first 15 trial dictionaries to training
            test_trials.extend(data[train_per_exp:train_per_exp + test_per_exp])  # Add next 5 trial dictionaries to testing
    
    print(f"Collected {len(train_trials)} training trials and {len(test_trials)} test trials")
    return train_trials, test_trials

def prepare_chisco_dataset(experiments=[10, 11, 12], train_per_exp=20, test_per_exp=10, seed=24573471, channels=122):
    # random.seed(seed)
    # np.random.seed(seed)
    if channels > 122:
        print(f"Channels must be less than 122")

    total_trials_to_use = train_per_exp + test_per_exp
    
    all_train_data = []  # List of (eeg_data, english_text) tuples for training
    all_test_data = []   # List of (eeg_data, english_text) tuples for testing
    
    # experiments = [10, 11, 12]
    for exp_num in experiments:  # exp_num = single experiment number
        print(f"Processing experiment {exp_num}...")
        
        # data = list of 130 trial dictionaries
        data = get_experiment(exp_num)
        if data is None:
            continue
            
        # trials_subset = list of first 30 trial dictionaries (where labels exist)
        trials_subset = data[:total_trials_to_use]
        
        processed_trials = []  # List of (eeg_array, english_text) tuples
        
        # trials_subset = list of trial dictionaries
        for trial in trials_subset:  # trial = single dictionary with 'text' and 'input_features'
            # Extract EEG data
            # eeg_raw = (1, 125, 1651) numpy array
            eeg_raw = trial['input_features']
            
            # eeg_processed = (122, 1651) numpy array in microvolts
            eeg_processed = eeg_raw[0, :channels, :] * 1000000 #data preproessing that official implementation in github did too
            
            # Get Chinese text and convert to English
            chinese_text = trial['text'].strip()  # chinese_text = string
            english_text = translation_dict.get(chinese_text, chinese_text)  # english_text = string
            
            # processed_trials = list of (eeg_array, english_text) tuples
            processed_trials.append((eeg_processed, english_text))
        
        # Randomly shuffle trials for this experiment
        # processed_trials = list of 30 (eeg_array, english_text) tuples, now shuffled
        random.shuffle(processed_trials)
        
        # Split into train/test
        # train_trials = list of 20 (eeg_array, english_text) tuples
        train_trials = processed_trials[:train_per_exp]
        # test_trials = list of 10 (eeg_array, english_text) tuples  
        test_trials = processed_trials[train_per_exp:train_per_exp + test_per_exp]
        
        # all_train_data = accumulated list of (eeg_array, english_text) tuples across experiments
        all_train_data.extend(train_trials)
        # all_test_data = accumulated list of (eeg_array, english_text) tuples across experiments
        all_test_data.extend(test_trials)
        
        print(f"  Added {len(train_trials)} training and {len(test_trials)} testing trials")
    
    print(f"\nTotal dataset: {len(all_train_data)} training, {len(all_test_data)} testing trials")
    
    # Final shuffle of combined data
    # all_train_data = list of (eeg_array, english_text) tuples, shuffled across all experiments
    random.shuffle(all_train_data)
    # all_test_data = list of (eeg_array, english_text) tuples, shuffled across all experiments
    random.shuffle(all_test_data)
    
    return all_train_data, all_test_data

#segment single eeg trial into fixed-size windows with padding for remainder
def segment_eeg_trial(eeg_data, window_size):
    #eeg_data shape: (channels=122, time_samples=1651)
    channels, total_samples = eeg_data.shape
    
    #calculate number of full windows and remainder
    num_full_windows = total_samples // window_size
    remainder_samples = total_samples % window_size
    
    windows = []
    
    #extract full windows
    for i in range(num_full_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window = eeg_data[:, start_idx:end_idx]  #shape: (channels, window_size)
        windows.append(window)
    
    #handle remainder samples if they exist
    if remainder_samples > 0:
        start_idx = num_full_windows * window_size
        remainder_window = eeg_data[:, start_idx:]  #shape: (channels, remainder_samples)
        
        #zero padding
        padding_needed = window_size - remainder_samples
        zero_padding = np.zeros((channels, padding_needed))
        padded_window = np.concatenate([remainder_window, zero_padding], axis=1)
        
        windows.append(padded_window)
    
    return windows  #list of arrays, each with shape (channels, window_size)

#convert eeg windows to ctm-compatible format and create torch tensors
def prepare_windows_for_ctm(windows):
    #windows: list of (channels, window_size) arrays
    #convert to: list of (window_size, channels) arrays for ctm attention
    ctm_windows = []
    
    for window in windows:
        #transpose from (channels, window_size) to (window_size, channels)
        ctm_window = window.transpose()  #shape: (window_size, channels)
        ctm_windows.append(ctm_window)
    
    #convert to torch tensors
    torch_windows = [torch.FloatTensor(window) for window in ctm_windows]
    
    return torch_windows  #list of tensors, each with shape (window_size, channels)

#prepare complete torch dataset with windowed eeg trials
#modified 
def prepare_torch_dataset(dataset, window_size, batch_size=None):
    #dataset: list of (eeg_array, english_text) tuples
    #returns: list of batch dictionaries for parallel processing
    
    print(f"preparing torch dataset with window_size={window_size}...")
    
    #process all trials into windowed format first
    all_trial_windows = []
    all_text_labels = []
    
    for trial_idx, (eeg_data, text_label) in enumerate(dataset):
        #segment eeg trial into windows
        windows = segment_eeg_trial(eeg_data, window_size)
        
        #convert windows to ctm format and torch tensors
        torch_windows = prepare_windows_for_ctm(windows)
        
        all_trial_windows.append(torch_windows)
        all_text_labels.append(text_label)
        
        if (trial_idx + 1) % 10 == 0:
            print(f"  processed {trial_idx + 1} trials...")
    
    #return individual trials if no batching requested
    if batch_size is None:
        return list(zip(all_trial_windows, all_text_labels))
    
    #randomly shuffle trials and texts together to maintain correspondence
    paired_data = list(zip(all_trial_windows, all_text_labels))
    random.shuffle(paired_data)
    
    #create parallel processing batches
    batches = []
    for i in range(0, len(paired_data), batch_size):
        batch_data = paired_data[i:i + batch_size]
        
        batch_trial_windows = []
        batch_texts = []
        
        for trial_windows, text_label in batch_data:
            batch_trial_windows.append(trial_windows)
            batch_texts.append(text_label)
        
        batch_dict = {
            'trial_windows': batch_trial_windows,
            'texts': batch_texts
        }
        
        batches.append(batch_dict)
    
    print(f"torch dataset prepared: {len(batches)} batches with {batch_size} trials each")
    return batches

#get dataset statistics for windowed data
def get_windowed_dataset_info(torch_dataset):
    print(f"windowed dataset statistics:")
    print(f"total trials: {len(torch_dataset)}")
    
    if len(torch_dataset) > 0:
        #analyse first trial
        first_windows, _ = torch_dataset[0]
        print(f"windows per trial (example): {len(first_windows)}")
        print(f"window shape: {first_windows[0].shape}")  #should be (window_size, channels)
        print(f"window tensor type: {first_windows[0].dtype}")
        
        #calculate average windows per trial
        total_windows = sum(len(windows) for windows, _ in torch_dataset)
        avg_windows = total_windows / len(torch_dataset)
        print(f"average windows per trial: {avg_windows:.1f}")
        
        print(f"sample labels:")
        for i, (_, text) in enumerate(torch_dataset[:3]):
            print(f"  {i+1}: {text}")

def plot_eeg_trial(eeg_data, english_text, channels_to_plot=5, figsize=(12, 8)):
    # eeg_data = (122, 1651) numpy array
    # time_axis = (1651,) numpy array representing 3.3 seconds
    time_axis = np.linspace(0, 3.3, eeg_data.shape[1])
    
    # Select channels to plot
    # channel_indices = list of integers for channel selection
    channel_indices = np.linspace(0, eeg_data.shape[0]-1, channels_to_plot, dtype=int)
    
    plt.figure(figsize=figsize)
    
    # channel_indices = list of channel numbers to plot
    for i, ch_idx in enumerate(channel_indices):  # ch_idx = single channel index
        plt.subplot(channels_to_plot, 1, i+1)
        # eeg_data[ch_idx, :] = (1651,) array of voltage values for one channel
        plt.plot(time_axis, eeg_data[ch_idx, :])
        plt.title(f'Channel {ch_idx}')
        plt.ylabel('µV')
        if i == channels_to_plot-1:
            plt.xlabel('Time (seconds)')
    
    plt.suptitle(f'EEG Data: "{english_text}"', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_eeg_heatmap(eeg_data, english_text, figsize=(15, 8)):
    # eeg_data = (122, 1651) numpy array
    plt.figure(figsize=figsize)
    plt.imshow(eeg_data, aspect='auto', cmap='RdBu_r', 
               extent=[0, 3.3, eeg_data.shape[0], 0])
    plt.colorbar(label='µV')
    plt.xlabel('Time (seconds)')
    plt.ylabel('EEG Channel')
    plt.title(f'EEG Heatmap: "{english_text}"')
    plt.show()

def get_dataset_info(dataset):
    # dataset = list of (eeg_array, english_text) tuples
    print(f"Dataset size: {len(dataset)} trials")
    if len(dataset) > 0:
        # eeg_shape = (122, 1651) - shape of single EEG trial
        eeg_shape = dataset[0][0].shape
        print(f"EEG data shape per trial: {eeg_shape}")
        print(f"Sample labels:")
        # dataset[:5] = first 5 (eeg_array, english_text) tuples
        for i, (_, text) in enumerate(dataset[:5]):  # text = english text string
            print(f"  {i+1}: {text}")

# if __name__ == "__main__":
#     # Prepare the dataset
#     # training = list of (eeg_array, english_text) tuples for training
#     # testing = list of (eeg_array, english_text) tuples for testing
#     training, testing = prepare_chisco_dataset()
    
#     # Print dataset information
#     print("\n=== TRAINING DATASET ===")
#     get_dataset_info(training)
    
#     print("\n=== TESTING DATASET ===")
#     get_dataset_info(testing)
    
#     # Plot example trial
#     if len(training) > 0:
#         print("\n=== PLOTTING EXAMPLE ===")
#         # training[0] = (eeg_array, english_text) tuple
#         # random_index = np.random.randint(len(training))
#         example_eeg, example_text = training[0]  # example_eeg = (122, 1651), example_text = string
#         plot_eeg_trial(example_eeg, example_text)
#         plot_eeg_heatmap(example_eeg, example_text)