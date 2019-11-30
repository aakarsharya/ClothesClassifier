# ClothesClassifier
Implemented a neural network that classifies different articles of clothing, only using Numpy to understand fundamentals.

## Installation
Use the pip package manager to install the following packages.
```bash
pip install numpy
pip install scipy
pip install -U scikit-learn
pip install matplotlib
```

## Running Python Files
Ensure you have python version >= 3.5 by entering the following in your terminal.
```bash
python --version
```
Clone the repository to your Desktop by entering the following commands.
```bash
cd ~/Desktop
git clone https://github.com/aakarsharya/ClothesClassifier.git
```
To run the files in this repository, you must first navigate to this directory from your terminal.
```bash
cd ClothesClassifier
```
To watch the neural network train itself, run nn.py from your terminal with the following command.
```bash
python nn.py
```
To test the neural network's accuracy, run test.py from your terminal with the following command.
```bash
python test.py
```
Remember to close the test image to allow the program to continue.

## How it Works
### Training the Neural Network
This program trains the neural network using thousands of clothes images, with labels that tell the network what article of clothing each image corresponds to. 

### Testing the Neural Network
This program allows you to test different images from a dataset that neural network has never seen before. The network uses its weights, that it has tuned by iterating through the training dataset, to classify these new images.