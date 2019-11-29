import pandas as pd
import numpy as np

trainData = pd.read_csv("datasets/dataset_train.csv")
testData = pd.read_csv("datasets/dataset_test.csv")

Xtrain = np.array(trainData.iloc[:, 1:])
Xtrain = np.transpose(Xtrain)

Ytrain = np.array(trainData.iloc[:,0])
Ytrain.shape = (60000,1)
Ytrain = np.transpose(Ytrain)
YtrainOneHot = np.eye(10)[Ytrain.astype('int32')]
YtrainOneHot = YtrainOneHot.T.reshape(10, 60000)

Xtest = np.array(testData.iloc[:,1:])
Xtest = np.transpose(Xtest)

Ytest = np.array(testData.iloc[:,0])
Ytest.shape = (10000,1)
Ytest = np.transpose(Ytest)
YtestOneHot = np.eye(10)[Ytest.astype('int32')]
YtestOneHot = YtestOneHot.T.reshape(10, 10000)

m = Xtrain.shape[1]
LEARNING_RATE = 1
EPOCHS = 2000
L = 3 # number of layers
costs = []
gradients = {}
activations = {}
weights = {}
layerSizes = [Xtrain.shape[0], 32, 16, 10] # size of each layer in neural network