import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import *

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def sigmoidDerivative(dA, Z):
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s) 
    return dZ

def softmaxDerivative(Y, AL):
    return AL-Y

def initializeWeights(layerSizes):
    for l in range(1,L+1):
        weights['W'+str(l)] = np.random.randn(layerSizes[l], layerSizes[l-1])
        weights['b'+str(l)] = np.zeros((layerSizes[l], 1))

def forwardProp(X, activations, weights):
    activations['A0'] = X
    for l in range(1,L):
        activations['Z'+str(l)] = np.dot(weights['W'+str(l)], activations['A'+str(l-1)]) + weights['b'+str(l)]
        activations['A'+str(l)] = sigmoid(activations['Z'+str(l)])
    activations['Z'+str(L)] = np.dot(weights['W'+str(L)], activations['A'+str(L-1)]) + weights['b'+str(L)]
    activations['A'+str(L)] = softmax(activations['Z'+str(L)])
    return activations['A'+str(L)]

def computeCost(Y, AL):
    loss = np.sum(np.multiply(Y, np.log(AL)))
    return (-1/m) * loss

def backProp(Y):
    gradients['dZ'+str(L)] = softmaxDerivative(Y, activations['A'+str(L)])
    gradients['dW'+str(L)] = (1/m) * np.dot(gradients['dZ'+str(L)], np.transpose(activations['A'+str(L-1)]))
    gradients['db'+str(L)] = (1/m) * np.sum(gradients['dZ'+str(L)], axis=1, keepdims=True)
    for l in reversed(range(1,L)):
        gradients['dA'+str(l)] = np.dot(np.transpose(weights['W'+str(l+1)]), gradients['dZ'+str(l+1)])
        gradients['dZ'+str(l)] = sigmoidDerivative(gradients['dA'+str(l)], activations['Z'+str(l)])
        gradients['dW'+str(l)] = (1/m) * np.dot(gradients['dZ'+str(l)], np.transpose(activations['A'+str(l-1)]))
        gradients['db'+str(l)] = (1/m) * np.sum(gradients['dZ'+str(l)], axis=1, keepdims=True)

def gradientDescent():
    for l in range(1,L+1):
        weights['W'+str(l)] -= LEARNING_RATE * gradients['dW'+str(l)]
        weights['b'+str(l)] -= LEARNING_RATE * gradients['db'+str(l)]

def plotTraining():
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(LEARNING_RATE))
    plt.show()

def storeWeights(layers):
    for l in range(1,layers+1):
        np.savetxt('trainedWeights/weights'+str(l)+'.txt', weights['W'+str(l)])
        np.savetxt('trainedWeights/bias'+str(l)+'.txt', weights['b'+str(l)])

def loadWeights(layers, layerSizes):
    loadedWeights = {}
    for l in range(1,layers+1):
        loadedWeights['W'+str(l)] = np.loadtxt('trainedWeights/weights'+str(l)+'.txt')
        loadedWeights['W'+str(l)].shape = (layerSizes[l], layerSizes[l-1])
        loadedWeights['b'+str(l)] = np.loadtxt('trainedWeights/bias'+str(l)+'.txt')
        loadedWeights['b'+str(l)].shape = (layerSizes[l], 1)
    return loadedWeights

def testNeuralNetwork(X_test, Y_test, loadedWeights):
    testActivations = {}
    AL = forwardProp(X_test, testActivations, loadedWeights)
    predictions = np.argmax(AL, axis=0)
    labels = np.argmax(Y_test, axis=0)
    # display results
    print('------------------------------')
    print('confusion matrix')
    print(confusion_matrix(predictions, labels))
    print('------------------------------')
    print('classification report')
    print(classification_report(predictions, labels))
    print('------------------------------')
    return predictions, labels

def train(X, Y, epochs):
    cost = 0
    for epoch in range(1, epochs):
        forwardProp(X, activations, weights)
        cost = computeCost(Y, activations['A'+str(L)])
        backProp(Y)
        gradientDescent()
        if (epoch % 100 == 0):
            costs.append(cost)
            print("Epoch " + str(epoch) + ": cost " + str(cost))
    print("Final cost after training: " + str(cost))

if __name__ == "__main__":
    initializeWeights(layerSizes)
    train(Xtrain, YtrainOneHot, EPOCHS
    )
    storeWeights(L)
    plotTraining()

