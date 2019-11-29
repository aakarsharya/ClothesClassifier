import numpy as np
import matplotlib.pyplot as plt
from preprocess import Xtest, YtestOneHot, L, layerSizes
from nn import loadWeights, testNeuralNetwork

weights = loadWeights(L, layerSizes)
predictions, labels = testNeuralNetwork(Xtest, YtestOneHot, weights)
clothes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

pixels = Xtest[:,9]
pixels = np.array(pixels, dtype=float)
image = pixels.reshape(28,28)
plt.imshow(image)
plt.show()
print('This network thinks this is a ' + str(clothes[predictions[9]]))
print('This is a ' + str(clothes[labels[9]]))
