from file_functions import *
from models_bachelors import *
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
from sklearn.metrics import accuracy_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def get_uncertainty(y_pred, unc_method):
    if unc_method == 'predictive-entropy':
        return predictive_uncertainty(y_pred, 'predictive-entropy')
    elif unc_method == 'mutual-information':
        return normalize_information(mutual_information(y_pred))
    elif unc_method == 'shannon-entropy':
        return predictive_uncertainty(y_pred, 'shannon-entropy')


def get_corrects(Y_true, Y_pred, axis):
    if not checkIfStandard(Y_pred):
        Y_pred = np.mean(Y_true, axis=-3)       # averages forward passes if not already averaged
    return np.argmax(Y_true, axis=axis) == np.argmax(Y_pred, axis=axis)

def load_predictions(n, flag):
    if flag == 'standard':
        return load_dict_from_hdf5(f'predictions/predictions_standard.h5')
    elif flag == 'ensemble_dropout':
        return load_dict_from_hdf5(f'predictions/predictions_ensemble_dropout.h5')
    else:
        return load_dict_from_hdf5(f'predictions/predictions_{n}.h5')

def avg_forward_passes(data):
    data["preds"] = data["preds"].mean(axis=-3)
    return data

'''
Get accuracies for each subject and ret as list
'''
def get_accuracies(data, isStandard):
    acc = []
    print(f'data shape: {data["preds"].shape}')
    data = avg_forward_passes(data) if not isStandard else data
    print(f'data shape: {data["preds"].shape}')
    y_preds = data["preds"].argmax(axis=-1)
    y_trues = data["labels"].argmax(axis=-1)
    
    # Get accuracy of each subject
    for idx, subject in enumerate(y_trues):
        print(idx, subject.shape)
        score = accuracy_score(y_pred=subject, y_true=y_preds[idx], normalize=True)
        acc.append(score)
    
    return acc

###########################################################################

'''The following is deprecated'''

###########################################################################
'''
takes as input a dict key: one of 'test' or 'lockbox'
'''
def avg_pred_entropy_plots(dataset, method, unc_method):
    bin_size = 0.05
    entropy_correct = []
    entropy_wrong = []
    N = 50

    # Iterate over all prediction sets. Also get the dropconnect predictions.
    for n in range(N):
        methods = load_predictions(n)
        data = methods[method]
        data = avg_forward_passes(data)
        if unc_method == 'predictive-normalised-entropy':
            entropy = predictive_uncertainty(data[dataset]['preds'])    # shape: (9,576)
        elif unc_method == 'mutual-information':
            entropy = mutual_information(data[dataset]['preds'])
        elif unc_method == 'shannon-entropy':
            entropy = shannon_entropy(data[dataset]['preds'])
        elif unc_method == 'predictive-entropy':
            entropy = predictive_entropy(data[dataset]['preds'])

        Y_true = data[dataset]['labels']    # shape: (9,576,4)
        corrects = get_corrects(Y_true, data[dataset]['preds'], axis=-1) # Get corrects across ALL subjects
        # Append the nth prediction's uncertainty estimations
        entropy_correct.append(entropy[corrects])
        entropy_wrong.append(entropy[~corrects])
        # For distribution plots of predictive entropy

    '''
    Check for data mismatch: entropy_correct is probably a list of np arrays instead of 
    1 cohesive np array 
    '''
    entropy_correct = np.hstack(entropy_correct)
    entropy_wrong = np.hstack(entropy_wrong)
    r = 5
    unc_cor = np.mean(entropy_correct)
    unc_cor_std = np.std(entropy_correct)
    unc_in = np.mean(entropy_wrong)
    unc_in_std = np.std(entropy_wrong)
    print(f"{dataset} avg. {unc_method} correct: {unc_cor:.5f} +/ {unc_cor_std:.5f}")
    print(f"{dataset} avg. {unc_method} wrong: {unc_in:.5f} +/ {unc_in_std:.5f}")

    # hist_data = [entropy_correct, entropy_wrong]    
    # group_labels = ['Correct', 'Incorrect']

    # # Normalizes AREA UNDER CURVE to sum up to 1. y-axis values are meaningless.
    # hist_correct, bins_correct, _ = plt.hist(entropy_correct, bins=20, density=True, alpha=0.5, label='Correct')
    # hist_wrong, bins_wrong, _ = plt.hist(entropy_wrong, bins=20, density=True, alpha=0.5, label='Wrong')
    # plt.legend()
    # # plt.show()

    # # Calculate overlap using histogram intersection
    # overlap = np.sum(np.minimum(hist_correct, hist_wrong))

    # # Normalize overlap between 0 and 1
    # normalized_overlap = overlap / np.sum(hist_correct)

    # print("Overlap:", normalized_overlap)
