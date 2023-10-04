import tensorflow as tf
from tensorflow import keras
from keras.constraints import max_norm
from keras_uncertainty.layers import DropConnectDense
from keras_uncertainty.utils import numpy_entropy

print(tf.__version__)

def uncertainty(probs):
  return numpy_entropy(probs, axis=-1)


def create_model():

      weights_filepath = 'TBD'

      C = 22          # Number of electrodes
      T = 1125        # Time samples of network input

      k_1 = 40        # K is number of convolutional kernels. SUBJECT TO HYPERPARAM TUNING
      f_1 = 25        # F is kernel size SUBJECT TO HYPERPARAM TUNING
      s_1 = (1,1)     # Strides size
      p_1 = (0,0)     # Padding size

      k_2 = 40
      f_2 = C
      s_2 = (1,1)
      p_2 = (0,0)

      m = 0.9
      alpha = 1
      f_p = (1, 75)   # Fp is pooling size
      s_p = (1, 15)   # Sp is pool stride

      Nc = 4          # Number of classes

      model = keras.models.Sequential([
              keras.layers.Conv2D(f_1,  k_1, padding = 'SAME',
                                  activation="linear",
                                  input_shape = (C, T, 1),
                                  kernel_constraint = max_norm(2)),
              keras.layers.Conv2D(f_2,  k_2, padding = 'SAME',
                                  input_shape = (1, C, T),
                                  activation="linear",
                                  kernel_constraint = max_norm(2)),

              keras.layers.BatchNormalization(momentum=0.9, epsilon=0.00001),
              keras.layers.ELU(alpha=1.0),
              keras.layers.AveragePooling2D(pool_size= f_p, strides= s_p),
              keras.layers.ELU(alpha=1.0),

              keras.layers.Flatten(),
              DropConnectDense(Nc, activation='softmax', prob=0.05, kernel_constraint = max_norm(0.5))
          ])



      # model.load_weights(weights_filepath)
      optimizer = keras.optimizers.Adam(learning_rate=1e-4)
      model.compile(loss="categorical_crossentropy",
                    optimizer=optimizer, metrics=["accuracy"])
      return model
