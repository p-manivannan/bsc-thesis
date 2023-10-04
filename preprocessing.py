from braindecode.datasets.moabb import MOABBDataset
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import (
    exponential_moving_standardize, preprocess, Preprocessor)
from numpy import multiply
from saver_loader import *
from sklearn.preprocessing import OneHotEncoder

def load_dataset():
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=None)
    return dataset

def preprocess_data(dataset):
    low_cut_hz = 4.  # low cut frequency for filtering
    high_cut_hz = 38.  # high cut frequency for filtering
    # Parameters for exponential moving standardization
    '''
    CHECK IF THE FACTOR IS SAME AS 0.999 MENTIONED IN
    THE ARTICLES
    '''
    factor_new = 1e-3
    init_block_size = 1000
    # Factor to convert from V to uV
    factor = 1e6
    iir_params = dict(order=3, ftype='butter', output='sos')

    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz, iir_params=iir_params, method='iir', phase='forward'),  # Third order butterworth filter
        # The logs say it's a causal filter but the order is 6?
        Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                    factor_new=factor_new, init_block_size=init_block_size)
    ]

    return preprocess(dataset, preprocessors)

# Expects a BaseConcatDataset object
def create_dataset(dataset):
  # Iterate through subject datasets
  lst = []
  for subject_id in range(0, len(dataset)):
  # Get trial datasets for subjects
    subject_dataset = dataset[subject_id].datasets
    # Iterate through trials
    for run in subject_dataset:
      # Get all epochs from the trial and add to list
      # A single trial (its first element) has shape (22, 1125)
      # Second element of the trial tuple is the class label as
      # an int. I have no idea what the third element is.
      # Some kind of array of three elements. First element is
      # always 0...
      for trial in run:
        lst.append(trial)

  # Get all runs in a single df.
  # Drop the weird (0, 625, 1750 column)
  df = pd.DataFrame(lst).drop(columns=2)
  targets = df.pop(1).squeeze()
  inputs = df[0].squeeze()

  return inputs.to_numpy(), targets.to_numpy()


def epoch_data(dataset):
    trial_start_offset_seconds = -0.5
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this. It needs parameters to
    # define how trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
    )

    return windows_dataset

def onehot(targets):
    encoder = OneHotEncoder(sparse=False)
    targets = targets.reshape(-1,1)
    targets = encoder.fit_transform(targets)
    return targets


def create_inputs_targets(windows_dataset):
    split_data = windows_dataset.split('subject')
    num_subjects = 9
    subject_dataset = [split_data[str(i)] for i in range(1, num_subjects + 1)]

    inputs, targets = create_dataset(subject_dataset)
    n_runs = len(inputs)
    channels = len(inputs[0])
    timestamps = len(inputs[0][0])
    inputs = np.vstack(inputs).reshape(n_runs, channels, timestamps)

    return inputs, targets


inputs, targets = create_inputs_targets(epoch_data(preprocess_data(load_dataset())))
targets = onehot(targets)
save_data('all_subject_runs', inputs, targets)
