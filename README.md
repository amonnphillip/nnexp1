# nnexp1
neural network experiment 1

This is a training exercise to help me learn more about neural nets. Written from scratch, this is a very, very simple implementation of a neural network. In this implementation I use a single type of neuron, sigmoid for all layers and train the network using the backpropagation method.

I train the network against a very small data set. The data simply trains the network to respond to (categorize) the input signal with a signal on one of its outputs. After the training the results of the tests are shown against each training data set. Error numbers close to 0 means the training was pretty successful.

To run you need to have nodejs installed (v4.2.4 +). From the command line type node index.js
