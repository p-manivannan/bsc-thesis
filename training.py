# DON'T FORGET TO USE A REGULAR MODEL WITHOUT UNCERTAINTY AS A BENCHMARK

from livelossplot import PlotLossesKeras
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras_uncertainty.models import StochasticClassifier
from model import create_model, uncertainty
from saver_loader import load_data
import tensorflow as tf

inputs, targets = load_data('all_subject_runs')

n_epochs= 1
num_subjects = 2
kfold = KFold(n_splits=num_subjects, shuffle=False)

fit_params = {"epochs: ", n_epochs}
early_stopping = EarlyStopping(monitor='loss', patience=3)

acc_per_fold = []
loss_per_fold = []

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

  X_train = inputs[train]
  X_test = inputs[test]
  Y_train = targets[train]
  Y_test = targets[test]

  # Define the model architecture
  model = create_model()
  model.fit(X_train, Y_train, epochs=n_epochs,
            callbacks=[early_stopping])

  mc_model = StochasticClassifier(model)
  unc_preds =  mc_model(X_test, num_samples=2)
  entropy = uncertainty(unc_preds)
  y_pred = tf.cast(tf.argmax(unc_preds, axis=1), tf.float32)
  y_pred = tf.argmax(y_pred, axis=0)
  print(y_pred)
  print(y_pred.shape)
  # Y_test = tf.argmax(Y_test, axis=0)

  acc = accuracy_score(Y_test, y_pred)
  print(f'Accuracy and Entropy for {fold_no}: {acc}')
  # # Increase fold number
  # fold_no = fold_no + 1
