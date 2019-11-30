import numpy as np
import matplotlib.pyplot as plt
from preprocess import Xtest, YtestOneHot
from nn import loadWeights, testNeuralNetwork

layerSizes = np.loadtxt('layerSizes.txt', dtype='int32')
weights = loadWeights(len(layerSizes)-1, layerSizes)
predictions, labels = testNeuralNetwork(len(layerSizes)-1, Xtest, YtestOneHot, weights)
clothes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
done = False
while (done == False):
    invalid = True
    while (invalid):
        userInput = input('Enter an index from 1-' + str(Xtest.shape[1]) + ': ')
        print('You entered: ' + str(userInput))
        userTest = int(userInput)-1
        if (userTest < 10000 and userTest >= 0):
            invalid = False
        else:
            print('Please enter a valid index.')
    pixels = Xtest[:,userTest]
    pixels = np.array(pixels, dtype=float)
    image = pixels.reshape(28,28)
    plt.imshow(image)
    plt.title("Test Image " + str(userInput))
    plt.show()
    print('The neural network thinks this is a ' + str(clothes[predictions[userTest]] + '.'))
    if (clothes[predictions[userTest]] == clothes[labels[userTest]]):
        print('Correct! This is a ' + str(clothes[labels[userTest]]) + '.')
    else:
        print('Incorrect. This is a ' + str(clothes[labels[userTest]]) + '.')
    finished = input('Try again? (y/n): ')
    done = finished == 'n'
