from standardmodels import DropoutModel
from keras.constraints import max_norm
from keras_uncertainty.models import DeepEnsembleClassifier

class EnsembleModel(DropoutModel):
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        super().__init__(hp, C, T, f, k1, fp, sp, Nc)

    # WIP
    # Q: How do I return an ensemble model when there is no tuned dropout?
    # Ans: Maybe check if the tuned dropout exists. If not, tune for dropout and set
    #      hyperparams of ensemble to dropout
    #      That's up to the tuning file to decide. That logic doesn't belong here
    
    def get_model(self, hp=None):
        if hp is not None:
            return DeepEnsembleClassifier(lambda: self.build(hp), num_estimators=10) 
        else:
            print('No hyperparameters provided! Either tune a standard model and retreive hyperparameters or specify a set of hyperparams (drop rates)')
            return None


