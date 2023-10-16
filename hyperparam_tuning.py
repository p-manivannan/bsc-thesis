from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from models_bachelors import *
from file_functions import *

'''
Shoddy ass script to do hyperparam tuning. I already managed
to tune for optimal drop_rates on jupyter notebook. This file
exists to make the training notebook cleaner.
'''

dataset = load('all_subject_runs')
loaded_inputs = dataset['inputs']
loaded_targets = dataset['targets']

n_epochs= 100
fit_params = {"epochs: ", n_epochs}
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
# There's an error in MCDropConnect so it's out of the list for now
methods = ['mcdropconnect']
subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
drop_rates = dict.fromkeys(methods, 0)

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
    drop_rates = tune_model(X_train, X_val, Y_train, Y_val, method)

mcdropout_tuner = kt.GridSearch(build_model_dropout,
                    objective='val_loss',
                    max_trials=100,
                    directory=f'mcdropout/hyperparams',
                    project_name='Bsc-thesis')

mcdropconnect_tuner = kt.GridSearch(build_model_connect,
                    objective='val_loss',
                    max_trials=100,
                    directory=f'mcdropconnect/hyperparams',
                    project_name='Bsc-thesis')

mcdropout_tuner.reload()
mcdropout_tuner.results_summary()
mcdropconnect_tuner.reload()
mcdropconnect_tuner.results_summary()