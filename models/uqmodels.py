from basemodel import BaseConvModel
from tensorflow import keras
from keras_uncertainty.layers import RBFClassifier, FlipoutDense
from keras_uncertainty.layers import duq_training_loop, add_gradient_penalty, add_l2_regularization
from keras.constraints import max_norm


class DUQModel(BaseConvModel):
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        super().__init__(hp, C, T, f, k1, fp, sp, Nc)

    def add_dense(self):
        self.model.add(keras.layers.Dense(self.hp.Choice('n_units_dense', [100, 200]), 
                                          activation='relu', kernel_constraint = max_norm(0.5)))
        return self
    
    def add_rbf_layer(self):
        centr_dims = self.hp.Choice('centroid_dims', [2, 5, 25, 100])
        length_scale = self.hp.Choice('length_scale', [0.1, 0.2, 0.3, 0.4, 0.5])
        train_centroids = self.hp.Choice('train_centroids', [False, True])
        self.model.add(RBFClassifier(self.Nc, length_scale, centroid_dims=centr_dims, trainable_centroids=train_centroids))
        return self

    def build(self, hp):
        self.hp = hp
        self.add_conv_filters()
        self.add_batch_norm()
        self.add_pooling()
        self.flatten()
        self.add_dense()
        self.add_rbf_layer()
        self.compile_model(loss='binary_crossentropy', metrics=["categorical_accuracy"])
        add_l2_regularization(self.model)
        return self.model

class FlipoutModel(BaseConvModel):
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        x_train_shape_0 = 3736                      # Set after inspecting training data
        num_batches = x_train_shape_0 / 32
        self.kl_weight = 1.0 / num_batches          # Param fixed during training
        super().__init__(hp, C, T, f, k1, fp, sp, Nc)

    def add_dense(self):
        self.model.add(keras.layers.Dense(self.hp.Choice('n_units_dense', [10, 25, 50]), 
                                          activation='relu', kernel_constraint = max_norm(0.5)))
        return self

    def add_flipout(self):
        prior_sigma_1 = self.hp.Choice('prior_sigma_1', [1.0, 2.5, 5.0])
        prior_sigma_2 = self.hp.Choice('prior_sigma_2', [1.0, 2.5, 5.0])
        prior_pi = self.hp.Choice('prior_pi', [0.1, 0.25, 0.5])
        n_units_2 = self.hp.Choice('n_units_2', [10, 25, 50])
        self.model.add(FlipoutDense(n_units_2, self.kl_weight, prior_sigma_1=prior_sigma_1, 
                                    prior_sigma_2=prior_sigma_2, prior_pi=prior_pi, 
                                    bias_distribution=False, activation='relu'))
        self.model.add(FlipoutDense(self.Nc, self.kl_weight, prior_sigma_1=prior_sigma_1, 
                                    prior_sigma_2=prior_sigma_2, prior_pi=prior_pi, 
                                    bias_distribution=False, activation='softmax'))
        return self
    
    def build(self, hp):
        self.hp = hp
        self.add_conv_filters()
        self.add_batch_norm()
        self.add_pooling()
        self.flatten()
        self.add_dense()
        self.add_flipout()
        self.compile_model()
        return self.model

