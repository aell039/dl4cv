import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # initialize the list of weights matrices, store network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # start looping from the index of the first layer but
        # stop before we reach the last two layers
        for i in np.arange(0, len(layers) - 2):
            # randomly initialize a weight matrix connecting all nodes in each layer and adding a bias node
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # the last two layers have the bias as an input but not as an output
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the network architecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        # compute the sigmoid activation value for a given input value
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # compute the derivative of the sigmoid function
        # Note: assumes x has already been passed through the sigmoid function
        return x * (1 - x)

    def fit(self, X, y, epochs=100, displayUpdate=100):
        # insert a column of 1s as the last entry in feature matrix
        # this allows us to treat bias as a trainable parameter
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop for desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over individual data point and train our network on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # check to see if we should display an update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        # construct out list of output activations for each layer as our datapoint flows
        # through the network. The first activation is just the input vector itself.
        A = [np.atleast_2d(x)]

        # FEEDFORWARD:
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by taking the dot product of
            # the activation and the weight matrix. This is the "net input"
            net = A[layer].dot(self.W[layer])

            # compute the "net output" by applying our activation function to the input
            out = self.sigmoid(net)

            # once we have the net output, add it to our list of activations
            A.append(out)

        # BACKPROPAGATION:
        # first compute the error (difference between prediction and actual)
        error = A[-1] - y

        # apply the chain rule and build our list of deltas 'D'
        # the first entry in the deltas is simply the error of the output layer times the
        # derivative of our activation function for the output value
        D = [error * self.sigmoid_deriv(A[-1])]

        # loop over the layers in reverse order - we have already accounted for the last two layers
        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the delta of the previous layer dotted
            # with the weight matrix of the current layer, followed by multiplying the delta by the
            # derivative of the nonlinear activation function for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # since we looped over the layers in reverse, we need to reverse the deltas
        D = D[::-1]

        # WEIGHT UPDATE:
        # loop over the layers
        for layer in np.arange(0, len(self.W)):
            # update weights by taking the dot product of the layer activations and their deltas
            # then multiplying this by the learning rate and adding to the wieght matrix
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        # initialize the output prediction as the input features that will pass through the network
        # to obtain final predictions
        p = np.atleast_2d(X)

        # add the bias column if required
        if addBias:
            # insert a column of 1s as the last entry of the feature matrix
            p = np.c_[p, np.ones((p.shape[0]))]

        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # compute the output the output by taking the dot product of the current activation
            # value 'p' and the layer's weight matrix, then passing through the activation function
            p = self.sigmoid(np.dot(p, self.W[layer]))

        # return the prediction
        return p

    def calculate_loss(self, X, targets):
        # make predictions for input then compute the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        # return the loss
        return loss
