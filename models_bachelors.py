from tensorflow import keras
import keras_tuner as kt
import numpy as np
import math
import keras.backend as K
from keras.constraints import max_norm
from keras_uncertainty.layers import DropConnectDense, StochasticDropout
from keras_uncertainty.utils import numpy_entropy

# need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   

def shanon_entropy(probs):
  return numpy_entropy(probs, axis=-1)

def predictive_entropy(samples):
  return -np.apply_along_axis(func1d=lambda x: x*np.log2(x), axis=-1, arr=samples).sum(axis=-1)

def mutual_information(samples):
  return predictive_entropy(samples) - shannon_entropy(samples)

def normalize_entropy(entropy, n_classes=4):
  return entropy / np.log2(n_classes)

def predictive_uncertainty(samples):
    entropy = predictive_entropy(samples)
    norm = normalize_entropy(entropy)
    return norm


def build_dropout_model(hp):

  # KERNEL SIZE IS HOW LONG SEGMENT IS THAT YOU'RE LOOKING AT. 
  # NUMBER OF FILTERS IS HOW MANY OF THESE SEGMENTS YOU'RE LEARNING IT IS ARBITRARY
      C = 22          # Number of electrodes
      T = 1125        # Time samples of network input

      f_1 = 40              # F is number of convolutional kernels.
      k_1 = (1, 25)         # K is kernel size 

      # One thats size of channels and another that's size of timestamps

      f_2 = 40
      k_2 = (C, 1)

      f_p = (1, 75)   # Fp is pooling size
      s_p = (1, 15)   # Sp is pool stride

      Nc = 4          # Number of classes
      drop_rates_1 = hp.Choice('drop_rates', [0.1, 0.2, 0.3, 0.4, 0.5])
      drop_rates_2 = hp.Choice('drop_rates', [0.1, 0.2, 0.3, 0.4, 0.5])
      conv_drop = hp.Boolean('conv_drop')
      # Another dropout layer before FC layer
      fc_drop = hp.Boolean('fc_drop')
      # I APPARENTLY SWITCH AROUND THE ORDER OF F1 AND K1
      # ADD THE AXIS PARAM FOR MAX NORM
      model = keras.models.Sequential()
      model.add(keras.layers.Conv2D(filters=f_1,  kernel_size=k_1, padding = 'SAME',
                                  activation="linear",
                                  input_shape = (C, T, 1),
                                  kernel_constraint = max_norm(2, axis=(0, 1, 2))))
      model.add(keras.layers.Conv2D(filters=f_2,  kernel_size=k_2, padding = 'SAME',
                                  activation="linear",
                                  kernel_constraint = max_norm(2, axis=(0, 1, 2))))
      if conv_drop:
        model.add(StochasticDropout(drop_rates_1))
      model.add(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-05))
      model.add(keras.layers.Activation(square))
      model.add(keras.layers.AveragePooling2D(pool_size= f_p, strides= s_p))
      model.add(keras.layers.Activation(log))
      if fc_drop:
        model.add(StochasticDropout(drop_rates_2))
      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(Nc, activation='softmax', kernel_constraint = max_norm(0.5)))

      optimizer = keras.optimizers.Adam(learning_rate=1e-4)
      model.compile(loss="categorical_crossentropy",
                    optimizer=optimizer, metrics=["accuracy"])
      return model

def build_dropconnect_model(hp):
  # KERNEL SIZE IS HOW LONG SEGMENT IS THAT YOU'RE LOOKING AT. 
  # NUMBER OF FILTERS IS HOW MANY OF THESE SEGMENTS YOU'RE LEARNING IT IS ARBITRARY
      C = 22          # Number of electrodes
      T = 1125        # Time samples of network input

      f_1 = 40        # K is number of convolutional kernels.
      k_1 = (1, 25)        # F is kernel size 

      # One thats size of channels and another that's size of timestamps

      f_2 = 40    # ITS SUPPOSED TO BE OTHER WAY AROUND. FILTER SIZE IS 
      k_2 = (C, 1)     # KERNEL SIZE SHOULD ACTUALLY BE C 

      f_p = (1, 75)   # Fp is pooling size
      s_p = (1, 15)   # Sp is pool stride

      Nc = 4          # Number of classes
      drop_rates_1 = hp.Choice('drop_rates', [0.1, 0.2, 0.3, 0.4, 0.5])
      drop_rates_2 = hp.Choice('drop_rates', [0.1, 0.2, 0.3, 0.4, 0.5])
      conv_drop = hp.Boolean('conv_drop')
      # Another dropout layer before FC layer
      fc_drop = hp.Boolean('fc_drop')
      # I APPARENTLY SWITCH AROUND THE ORDER OF F1 AND K1
      # ADD THE AXIS PARAM FOR MAX NORM
      model = keras.models.Sequential()
      model.add(keras.layers.Conv2D(filters=f_1,  kernel_size=k_1, padding = 'SAME',
                                  activation="linear",
                                  input_shape = (C, T, 1),
                                  kernel_constraint = max_norm(2, axis=(0, 1, 2))))
      model.add(keras.layers.Conv2D(filters=f_2,  kernel_size=k_2, padding = 'SAME',
                                  activation="linear",
                                  kernel_constraint = max_norm(2, axis=(0, 1, 2))))
      if conv_drop:
        model.add(DropConnectDense(22, drop_rates_1))
      model.add(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-05))
      model.add(keras.layers.Activation(square))
      model.add(keras.layers.AveragePooling2D(pool_size= f_p, strides= s_p))
      model.add(keras.layers.Activation(log))
      if fc_drop:
        model.add(DropConnectDense(22, drop_rates_2))
      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(Nc, activation='softmax', kernel_constraint = max_norm(0.5)))

      optimizer = keras.optimizers.Adam(learning_rate=1e-4)
      model.compile(loss="categorical_crossentropy",
                    optimizer=optimizer, metrics=["accuracy"])
      return model

def tune_model(x_train, x_val, y_train, y_val, method, callbacks):
    methods = {'mcdropout': build_dropout_model, 'mcdropconnect': build_dropconnect_model}
    tuner = kt.GridSearch(hypermodel=methods[method],
                          objective='val_loss',
                          max_trials=100,
                          executions_per_trial=1,
                          overwrite=True,
                          directory=f'{method}/tuning',
                          project_name=f'{method}')
    tuner.search(x_train, y_train, epochs=100, validation_data=(x_val, y_val),
                 callbacks=callbacks)
    return tuner

def build_standard_model(hp):
# KERNEL SIZE IS HOW LONG SEGMENT IS THAT YOU'RE LOOKING AT. 
# NUMBER OF FILTERS IS HOW MANY OF THESE SEGMENTS YOU'RE LEARNING IT IS ARBITRARY
    C = 22          # Number of electrodes
    T = 1125        # Time samples of network input

    f_1 = 40        # K is number of convolutional kernels.
    k_1 = (1, 25)        # F is kernel size 

    # One thats size of channels and another that's size of timestamps

    f_2 = 40    # ITS SUPPOSED TO BE OTHER WAY AROUND. FILTER SIZE IS 
    k_2 = (C, 1)     # KERNEL SIZE SHOULD ACTUALLY BE C 

    f_p = (1, 75)   # Fp is pooling size
    s_p = (1, 15)   # Sp is pool stride

    Nc = 4          # Number of classes
    drop_rates_1 = hp.Choice('drop_rates', [0.1, 0.2, 0.3, 0.4, 0.5])
    drop_rates_2 = hp.Choice('drop_rates', [0.1, 0.2, 0.3, 0.4, 0.5])
    conv_drop = hp.Boolean('conv_drop')
    # Another dropout layer before FC layer
    fc_drop = hp.Boolean('fc_drop')
    # I APPARENTLY SWITCH AROUND THE ORDER OF F1 AND K1
    # ADD THE AXIS PARAM FOR MAX NORM
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=f_1,  kernel_size=k_1, padding = 'SAME',
                                activation="linear",
                                input_shape = (C, T, 1),
                                kernel_constraint = max_norm(2, axis=(0, 1, 2))))
    model.add(keras.layers.Conv2D(filters=f_2,  kernel_size=k_2, padding = 'SAME',
                                activation="linear",
                                kernel_constraint = max_norm(2, axis=(0, 1, 2))))
    if conv_drop:
      model.add(keras.layers.Dropout(drop_rates_1))
    model.add(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-05))
    model.add(keras.layers.Activation(square))
    model.add(keras.layers.AveragePooling2D(pool_size= f_p, strides= s_p))
    model.add(keras.layers.Activation(log))
    if fc_drop:
      model.add(keras.layers.Dropout(drop_rates_2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(Nc, activation='softmax', kernel_constraint = max_norm(0.5)))

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])
    return model

'''
Returns the best tuned models. 
For MCDropout this is:
{'drop_rates': 0.4, 'conv_drop': False, 'fc_drop': True}
For MCDropConnect this is:
{'drop_rates': 0.2, 'conv_drop': True, 'fc_drop': False}
Top models for MCDropConnect all do not have UQ layers.
So the model description above corresponds to first
trial that had a UQ layer with val loss 0.822
'''
def load_tuned_models():
  mcdropout_tuner = kt.GridSearch(build_dropout_model,
                      objective='val_loss',
                      max_trials=100,
                      directory=f'mcdropout/tuning',
                      project_name=f'mcdropout')

  mcdropconnect_tuner = kt.GridSearch(build_dropconnect_model,
                      objective='val_loss',
                      max_trials=100,
                      directory=f'mcdropconnect/tuning',
                      project_name=f'mcdropconnect')

  mcdropout_tuner.reload()
  mcdropconnect_tuner.reload()
  # Dropout best params were index 0: 0.2 with only fc_drop
  # Dropconnect best params were index 5: 0.1 with only conv_drop
  # mcdropconnect_tuner.results_summary()
  dropout_best_hps = mcdropout_tuner.get_best_hyperparameters(num_trials=10)[0] # Top trial with any UQ layer
  dropconnect_best_hps = mcdropconnect_tuner.get_best_hyperparameters(num_trials=10)[5] # Top trial with any UQ layer
  return dropout_best_hps, dropconnect_best_hps
