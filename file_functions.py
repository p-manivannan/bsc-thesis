import h5py
import os, os.path
import numpy as np

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')

'''
Expects a dictionary for second arg. Dictionary format:
{'name': data, ....}
'''
def save(filename, dict):
  file = h5py.File(filename + '.h5', 'w')
  for name in dict:
     file.create_dataset(name, data=dict[name])
  file.close()


def load(filename):
  file = h5py.File(filename+'.h5', 'r')
  dictionary = {}
  for keys in file:
    dictionary[keys] = file[keys][:]
        
  file.close()
  return dictionary

# This function is used to to reshape np array shaped like
# (num_subjects, num_trials, num_channels, num_timestamps)
# for inputs and
# (num_subjects, num_trials, num_classes) for targets.
# to (total_num_trials, num_channels, num_timestamps)
# for input and you same for targets: Total trials in 
# first dimension.
def get_x_y(inputs, targets):
    n_subjects = inputs.shape[0]
    n_runs = inputs.shape[1] * n_subjects
    channels = inputs.shape[2]
    timestamps = inputs.shape[3]
    n_classes = targets.shape[2]
    X = np.vstack(inputs).reshape(n_runs, channels, timestamps)
    Y = np.vstack(targets).reshape(n_runs, n_classes)
    return X, Y