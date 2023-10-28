import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from livelossplot import PlotLossesKeras
from models_bachelors import *
from file_functions import *
import tensorflow as tf
import keras_tuner as kt
dataset = load('all_subject_runs_no_preprocess')
lockbox = load('lockbox')['data']
loaded_inputs = dataset['inputs']
loaded_targets = dataset['targets']


n_epochs= 100
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
callbacks = [early_stopping]
# There's an error in MCDropConnect so it's out of the list for now
methods = ['mcdropout', 'mcdropconnect']
subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Training loop for MCDropout and MCDropconnect models
for method in methods:
    # This loop leaves one subject for testing (denoted by the number in the name of the weights file).
    # Then it combines all the subject trials such that shape is now (8 * 576, 22, 1125).
    # Then selects 10% of this as the validation set.
    subject_id = 0
    train_ids = subject_ids[:]
    train_ids.remove(subject_id)
    train_inputs = np.vstack(loaded_inputs[train_ids])
    train_targets = np.vstack(loaded_targets[train_ids])        
    X_train, X_val, Y_train, Y_val = train_test_split(train_inputs, train_targets, test_size=0.1)
    # Do hyperparam tuning (SUBJECT 0)
    best_model = tune_model(X_train, X_val, Y_train, Y_val, method, callbacks)