import math
import copy
import torch
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
import sklearn.preprocessing
# def plot_accuracies(hist):
#     print("Final Accuracy: " + str(hist.history['accuracy'][len(hist.history['accuracy'])-1]))
#     plt.plot(hist.history['accuracy'])
#     plt.plot(hist.history['val_accuracy'])
#     if len(hist.history['accuracy'])==1:
#         plt.plot([0], [hist.history['accuracy'][0]], marker='o', markersize=3, color="blue")
#         plt.plot([0], [hist.history['val_accuracy'][0]], marker='o', markersize=3, color="orange")

#     plt.title('Model Accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Val'])
#     plt.show()
def runTraining(model, x_train, y_train, batchSize, epochs, learningRate):
  opt = RMSprop(learning_rate= learningRate)
  model.compile(optimizer = opt, loss= 'categorical_crossentropy', metrics = ['accuracy'])
  # model.summary()
  print(np.shape(x_train), np.shape(y_train))
  history = model.fit(x_train, y_train,
                    batch_size = batchSize,
                    epochs = epochs, validation_split = .2, verbose = 0)
#   plot_accuracies(history)
  return history.history['accuracy'][len(history.history['accuracy'])-1]

# def setup_():
#   model = Sequential()
#   model.add(Dense(xtrain.shape[1], activation='relu', input_shape=(xtrain.shape[1],)))
#   model.add(Dense(int(xtrain.shape[1]/2), activation='relu'))
#   model.add(Dense(len(np.unique(y_test)), activation="softmax"))

  
#   return model
def fit(model,xtrain,ytrain,batchsize,epochs,learningrate):
  ohe = sklearn.preprocessing.OneHotEncoder()
  transformed = ohe.fit_transform(np.expand_dims(ytrain,1))
  runTraining(model, xtrain, transformed.toarray(), batchsize, epochs, learningrate)
# model=setup_()
# fit(model,xtrain,ytrain,batchsize,epochs,learningrate)


##Requires Numpy Input

class MLP():
    def __init__(self, classes : int, features : int, dim : int = 400):
        model = Sequential()
        model.add(Dense(dim, activation='relu',input_shape=(features,)))
        model.add(Dense(classes, activation="softmax"))
        self.model=model
    def fit(self,xtrain,ytrain,batch_size,epochs,lr):
        fit(self.model,xtrain,ytrain,batch_size,epochs,lr)
    def __call__(self,xtest):
        yhat = self.model.predict(xtest)
        return np.array([row.argmax() for row in yhat])