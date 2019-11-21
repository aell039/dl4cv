import cv2


class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target dimensions and Interpolation method
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize image to fixed size, ignoring aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
