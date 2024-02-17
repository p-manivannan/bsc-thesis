from basemodel import BaseConvModel
from keras_uncertainty.layers import DropConnectDense, StochasticDropout
import keras_tuner as kt
from keras.constraints import max_norm

class MCDropConnectModel(BaseConvModel):
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        super().__init__(hp, C, T, f, k1, fp, sp, Nc)

    def add_dropconnect(self):
        self.model.add(DropConnectDense(22, 
                                        self.hp.Choice('drop_rates', 
                                                       [0.1, 0.2, 0.3, 0.4, 0.5])))
        return self
    
    def build(self, hp):
        self.hp = hp
        fc_drop = self.hp.Boolean('fc_drop')
        conv_drop = self.hp.Boolean('conv_drop')
        self.add_conv_filters()
        self.add_dropconnect() if conv_drop else None
        self.add_batch_norm()
        self.add_pooling()
        self.add_dropconnect() if fc_drop else None
        self.flatten()
        self.add_dense()
        self.compile_model()
        return self.model

class MCDropoutModel(BaseConvModel):
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        super().__init__(hp, C, T, f, k1, fp, sp, Nc)

    def add_dropout(self):
        self.model.add(StochasticDropout(self.hp.Choice('drop_rates',
                                                        [0.1, 0.2, 0.3, 0.4, 0.5])))
        return self
        
    def build(self, hp):
        self.hp = hp
        fc_drop = self.hp.Boolean('fc_drop')
        conv_drop = self.hp.Boolean('conv_drop')
        self.add_conv_filters()
        self.add_dropout() if conv_drop else None
        self.add_batch_norm()
        self.add_pooling()
        self.add_dropout() if fc_drop else None
        self.flatten()
        self.add_dense()
        self.compile_model()
        return self.model