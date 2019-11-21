from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import numpy as np
import argparse


def sigmoid_acivation(x):
    # compute the sigmoid activation value for a given input
    return 1.0 / (1 + np.exp(-x))


def predict(X, W):
    # take the dot product between our features and weight matrix
    preds = sigmoid_acivation(X.dot(W))

    # apply a step function to threshold the outputs to binary class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds


def next_batch(X, y, batchSize):
    # loop over dataset X in mini batches
    # yield a tuple of current batch data and labels
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i : i + batchSize], y[i : i + batchSize])


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="number of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
ap.add_argument(
    "-b", "--batchsize", type=int, default=32, help="size of SGD mini-batches"
)
args = vars(ap.parse_args())


# generate a 2 class classification problem with 1000 data points, where
# each data point is a 2d feature vector
(X, y) = make_blobs(
    n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1
)
y = y.reshape((y.shape[0], 1))

# insert a column of 1s at the end of the feature matrix
# this is to allow us to put the bias in the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

# split into train and test sets, 50% each
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# initialise weight matrix and list of losses
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

# loop for desired number of epochs
for epoch in np.arange(0, args["epochs"]):
    # initialise the total loss for the epoch
    epochLoss = []

    # loop over the data in batches
    for (batchX, batchY) in next_batch(X, y, args["batchsize"]):
        # take the dot product of features (X) and weights (W), then
        # pass through sigmoid activation to get predictions
        preds = sigmoid_acivation(batchX.dot(W))

        # calculate error
        error = preds - batchY
        epochLoss.append(np.sum(error ** 2))

        # the gradient descent update is the dot product between the
        # features and the error
        gradient = batchX.T.dot(error)

        # nudge the weights in the direction of the gradient by a step
        # the size of the learning rate
        W += -args["alpha"] * gradient

    # update our loss history with the average loss across all batches
    loss = np.average(epochLoss)
    losses.append(loss)

    # print updates
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

# evaluate the model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))
