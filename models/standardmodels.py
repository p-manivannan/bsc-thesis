from basemodel import BaseConvModel
import dropconnect_tensorflow as dc
from keras.constraints import max_norm

class DropConnectModel(BaseConvModel):
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        super().__init__(hp, C, T, f, k1, fp, sp, Nc)

    def add_dropconnect(self):
        self.model.add(dc.DropConnectDense(units=22, prob=self.hp.Choice('drop_rates',
                                           [0.1, 0.2, 0.3, 0.4, 0.5]), 
                                           activation="linear", use_bias=False))
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


class DropoutModel(BaseConvModel):
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        super().__init__(hp, C, T, f, k1, fp, sp, Nc)

    def add_dropout(self):
        self.model.add(keras.layers.Dropout(self.hp.Choice('drop_rates', 
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

