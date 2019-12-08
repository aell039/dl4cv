from pyimagesearch.nn import Perceptron
import numpy as np

# construct an XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# define our perceptron and train it
print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

# now that the network is trained we can test it
print("[INFO] testing perceptron...")

# loop over data points
for (x, target) in zip(X, y):
    # make a prediction and print it
    pred = p.predict(x)
    print("[INFO] data={}, ground-truth={}, prediction={}".format(x, target[0], pred))
