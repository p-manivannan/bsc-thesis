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

def dict2hdf5(filename, dic):
    with h5py.File(filename, 'w') as h5file:
        recursive_dict2hdf5(h5file, '/', dic)


def recursive_dict2hdf5(h5file, path, dic):
    for key, item in dic.items():
        if not isinstance(key, str):
            key = str(key)
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, list):
            h5file[path + key] = np.array(item)
        elif isinstance(item, dict):
            recursive_dict2hdf5(h5file, path + key + '/',
                                item)
        else:
            raise ValueError('Cannot save %s type' % type(item))
        
def save_dict_to_hdf5(dic, filename):

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def load_dict_from_hdf5(filename):

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')



def recursively_save_dict_contents_to_group( h5file, path, dic):

    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")        

    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")
    # save items to the hdf5 file
    for key, item in dic.items():
        #print(key,item)
        key = str(key)
        if isinstance(item, list):
            item = np.array(item)
            #print(item)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # save strings, numpy.int64, and numpy.float64 types
        if isinstance(item, (np.int64, np.float64, str, np.float, float, np.float32,int)):
            #print( 'here' )
            h5file[path + key] = item
            if not h5file[path + key].value == item:
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save numpy arrays
        elif isinstance(item, np.ndarray):            
            try:
                h5file[path + key] = item
            except:
                item = np.array(item).astype('|S9')
                h5file[path + key] = item
            if not np.array_equal(h5file[path + key].value, item):
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save dictionaries
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        # other types cannot be saved and will result in an error
        else:
            #print(item)
            raise ValueError('Cannot save %s type.' % type(item))

def recursively_load_dict_contents_from_group( h5file, path): 

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans            

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