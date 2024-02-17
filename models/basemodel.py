from tensorflow import keras
import keras.backend as K

class BaseConvModel:
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        self.C = C                               # Numbear of electrodes
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
    
    @staticmethod
    def square(x):
        return K.square(x)
    
    @staticmethod
    def log(x):
        return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))
    
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
        return self
    
    def add_batch_norm(self):
        self.model.add(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-05))
        self.model.add(keras.layers.Activation(lambda x: K.square(x)))
        return self

    def add_pooling(self):
        self.model.add(keras.layers.AveragePooling2D(pool_size= self.fp, 
                                                     strides= self.sp))
        self.model.add(keras.layers.Activation(lambda x: K.log(K.clip(x, min_value = 1e-7, max_value = 10000))))
        return self

    def flatten(self):
        self.model.add(keras.layers.Flatten())
        return self

    def add_dense(self):
        self.model.add(keras.layers.Dense(self.nc, activation='softmax', 
                                          kernel_constraint = max_norm(0.5)))
        return self
        
    def compile_model(self, lr=1e-4, loss='categorical_crossentropy', metrics=['accuracy']):
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics)
        return self

    def build(self):
        self.add_conv_filters()
        self.add_batch_norm()
        self.add_pooling()
        self.flatten()
        self.add_dense()
        self.compile_model()
        return self.model
    
    def get_model(self):
        return self.model
