from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

# construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# get the list of image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialise the preprocessor, load dataset, and reshape data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=1000)
data = data.reshape((data.shape[0], 3072))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition into train (75%) and test (25%) sets
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=5
)

# loop over our set of regularizers
for r in (None, "l1", "l2"):
    # train a SGD classifier using the softmax loss function and the specified
    # regularizer for 10 epochs
    print("[INFO] training model with '{}' penalty".format(r))
    model = SGDClassifier(
        loss="log",
        penalty=r,
        max_iter=10,
        learning_rate="constant",
        eta0=0.01,
        random_state=42,
    )
    model.fit(trainX, trainY)

    # evaluate the classifier
    acc = model.score(testX, testY)
    print("[INFO] '{}' penalty accuracy: {:.2f}%".format(r, acc * 100))
