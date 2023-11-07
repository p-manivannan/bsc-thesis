import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from file_functions import *
from models_bachelors import *
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
from keras_uncertainty.models import StochasticClassifier
from multiprocessing import Pool


def _apply_df(args):
    model, data = args
    return  model.predict(data, num_samples=50, batch_size=100)


def apply_by_multiprocessing(data, model, workers):
    pool = Pool(processes=workers)
    result = pool.map(_apply_df, (model, np.array_split(data, workers)))
    pool.close()
    return list(result)

def main():
    dataset = load('all_subject_runs_no_preprocess')
    loaded_inputs = dataset['inputs']
    loaded_targets = dataset['targets']
    '''
    Loads a dictionary with 2 keys: 'inputs', 'targets'. 
    Both keys have ndarray containing inputs and targets
    of 9 subjects separated by subject.
    '''
    lockbox = load('lockbox')['data']

    predictions = {'mcdropout': 
                {'test': {'preds':[], 'labels':[]}, 
                    'lockbox':{'preds':[], 'labels':[]}},
                    'mcdropconnect': 
                {'test': {'preds':[], 'labels':[]}, 
                    'lockbox':{'preds':[], 'labels':[]}},
                    'standard':
                {'test': {'preds':[], 'labels':[]}, 
                    'lockbox':{'preds':[], 'labels':[]}}
                }


    '''
    Load best hyperparams
    '''
    dropout_best_hps, dropconnect_best_hps = load_tuned_models()
    subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    N = 20


    # For each method, get preds and labels for each test subject
    # and their corresponding lockbox set.
    for method, values in predictions.items():
        print(f'{method}')
        if method == 'standard':
            wts_directory = f'mcdropout/weights'
        else:
            wts_directory = f'{method}/weights'
        # Iterate through test subjects
        for test_subject_id in range(0, 9):
            print(f'test subject {test_subject_id}')
            train_subj_ids = [x for x in subject_ids if x != test_subject_id]
            X_test = loaded_inputs[test_subject_id]
            Y_true = loaded_targets[test_subject_id]
            # Train set is sent in because lockbox is returned from the train set not the whole dataset.
            # This is because lockbox shape: (9, 8, 57) and inputs shape: (9, 576, 22, 1125)
            # Axis 0 are test_subj_ids and axis 1 are the train_subject_ids.
            # The function assumes that shape[0] of lockbox[test_subj_id] and shape[0] of
            # inputs is the same: 8.
            X_lock, Y_lock = get_lockbox_data(loaded_inputs[train_subj_ids], loaded_targets[train_subj_ids], lockbox[test_subject_id])
            wts_path = checkpoint_path = f'{wts_directory}/test_subject_{test_subject_id}.ckpt'
            if method == 'mcdropout':
                model = build_dropout_model(dropout_best_hps)
            elif method == 'mcdropconnect':
                model = build_dropconnect_model(dropconnect_best_hps)
            else:
                model = build_standard_model(dropout_best_hps)
            
            model.load_weights(wts_path)
            # Get Y_preds for test subject
            if method != 'standard':
                model = StochasticClassifier(model)
                Y_preds = apply_by_multiprocessing(data=X_test, model=model, workers=3)
                # Get lockboxed Y_preds for test subject
                lockbox_Y_preds = apply_by_multiprocessing(data=X_test, model=model, workers=3)
            else:
                Y_preds = model(X_test, batch_size=100, training=False)
                # Get lockboxed Y_preds for test subject
                lockbox_Y_preds = model(X_lock, batch_size=100, training=False)

            lockbox_Y_true = Y_lock
            values['test']['preds'].append(Y_preds)
            values['test']['labels'].append(Y_true)
            values['lockbox']['preds'].append(lockbox_Y_preds)
            values['lockbox']['labels'].append(lockbox_Y_true)

    for method, values in predictions.items():
        values['test']['preds'] = np.array(values['test']['preds'])
        values['test']['labels'] = np.array(values['test']['labels'])
        values['lockbox']['test'] = np.array(values['lockbox']['test'])
        values['lockbox']['labels'] = np.array(values['lockbox']['labels'])

    dict2hdf5('20_predictions.h5', predictions)


if __name__ == "__main__":
    main()
