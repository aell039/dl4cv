import numpy as np


class Perceptron:
    def __init__(self, N, alpha=0.1):
        # initialise weight matrix and store learning rate
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        # apply the step function
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # add a column of 1s to the feature matrix to allow us train on the bias too
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop for desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point
            for (x, target) in zip(X, y):
                # take the dot product of the input features and weight matrix
                # then apply step function to get the prediction
                p = self.step(np.dot(x, self.W))

                # only perform a weight update if our prediction does not match target
                if p != target:
                    # determine the error
                    error = p - target

                    # update the weight matrix
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        # ensure output is a matrix
        X = np.atleast_2d(X)

        # check to see if bias column should be added
        if addBias:
            # insert 1s as last value in feature matrix
            X = np.c_[X, np.ones((X.shape[0]))]

        # take the dot product of input features and weight matrix
        # then apply step function to get the prediction
        return self.step(np.dot(X, self.W))
