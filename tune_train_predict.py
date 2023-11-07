import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tuning import *
from training import *
from file_functions import *

'''
Load data
'''
dataset = load('all_subject_runs_no_preprocess')
lockbox = load('lockbox')['data']
loaded_inputs = dataset['inputs']
loaded_targets = dataset['targets']

# call_tuning_file(dataset, lockbox, loaded_inputs, loaded_targets)

call_training_file(dataset, lockbox, loaded_inputs, loaded_targets)

