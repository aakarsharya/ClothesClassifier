import numpy as np
import matplotlib.pyplot as plt
from preprocess import Xtest, YtestOneHot, L, layerSizes
from nn import loadWeights, testNeuralNetwork

weights = loadWeights(L, layerSizes)
predictions, labels = testNeuralNetwork(Xtest, YtestOneHot, weights)
clothes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
invalid = True
while (invalid):
    userInput = input('Enter an index from 1-' + str(Xtest.shape[1]) + ': ')
    print('you entered: ' + str(userInput))
    userTest = int(userInput)-1
    if (userTest < 10000 and userTest >= 0):
        invalid = False
    else:
        print('please enter a valid index.')
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
