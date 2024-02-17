from tensorflow import keras
import keras_tuner as kt
import numpy as np
import math
import keras.backend as K
from keras.constraints import max_norm
from keras_uncertainty.layers import DropConnectDense, StochasticDropout, RBFClassifier, FlipoutDense, FlipoutConv2D
from keras_uncertainty.layers import duq_training_loop, add_gradient_penalty, add_l2_regularization
from keras_uncertainty.models import DeepEnsembleClassifier
import dropconnect_tensorflow as dc


class BaseConvModel:
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        self.C = C                               # Number of electrodes
        self.T = T                               # Time samples of network input
        self.f = f                               # Number of convolutional kernels
        self.k1 = k1                             # Kernel size
        self.k2 = (self.C, 1)                    # Kernel size
        self.fp = (1, 75)                        # Pooling size
        self.sp = (1, 15)                        # Pool stride
        self.Nc = Nc                             # Number of classes
        self.input_shape = (self.C, self.T, 1)
        self.hp = hp
        self.model = keras.Models.Sequential()
    
    def add_conv_filters(self):
        self.model.add(keras.layers.Conv2D(filters=self.f,  kernel_size=self.k1, 
                                           padding = 'SAME',
                                           activation="linear",
                                           input_shape = self.input_shape,
                                           kernel_constraint = max_norm(2, axis=(0, 1, 2))))
        self.model.add(keras.layers.Conv2D(filters=self.f,  kernel_size=self.k2, 
                                           padding = 'SAME',
                                           activation="linear",
                                           kernel_constraint = max_norm(2, axis=(0, 1, 2))))

    def 