import pandas as pd
import numpy as np

C = 10 # number of classes (different types of clothing)
trainData = pd.read_csv("datasets/dataset_train.csv")
testData = pd.read_csv("datasets/dataset_test.csv")

Xtrain = np.array(trainData.iloc[:, 1:])
Xtrain = np.transpose(Xtrain)

Ytrain = np.array(trainData.iloc[:,0])
Ytrain.shape = (60000,1)
Ytrain = np.transpose(Ytrain)
YtrainOneHot = np.eye(C)[Ytrain.astype('int32')]
YtrainOneHot = YtrainOneHot.T.reshape(C, 60000)

Xtest = np.array(testData.iloc[:,1:])
Xtest = np.transpose(Xtest)

Ytest = np.array(testData.iloc[:,0])
Ytest.shape = (10000,1)
Ytest = np.transpose(Ytest)
YtestOneHot = np.eye(C)[Ytest.astype('int32')]
YtestOneHot = YtestOneHot.T.reshape(C, 10000)

layerSizes = np.array([Xtrain.shape[0], 64, 32, 16, C])
m = Xtrain.shape[1]
LEARNING_RATE = 1
EPOCHS = 2000
costs = []
gradients = {}
activations = {}
weights = {}

if __name__ == "__main__":
    np.savetxt('layerSizes.txt', layerSizes)