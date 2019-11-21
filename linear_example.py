import numpy as np
import cv2

# initialise class labels and set random number seed
labels = ["dog", "cat", "panda"]
np.random.seed(1)

# randomly initialise weight matrix and bias vector
W = np.random.randn(3, 3072)
b = np.random.randn(3)

# load a sample image, resize, and flatten
orig = cv2.imread("pup.jpg")
image = cv2.resize(orig, (32, 32)).flatten()

# compute output score
# dot product of weight matrix and feature vector plus bias
scores = W.dot(image) + b

# loop over scores and labels and display them
for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

# draw label with highest score on image
cv2.putText(
    orig,
    "Label: {}".format(labels[np.argmax(scores)]),
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 255, 0),
    2,
)

# display the image
cv2.imshow("Image", orig)
cv2.waitKey(0)
