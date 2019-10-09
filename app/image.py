import cv2


class Image:
    width = 0
    height = 0
    image = None

    def __init__(self, path):
        self.image = cv2.imread(path)
        [self.height, self.width, _] = self.image.shape

    def get_input_blob(self, blob_size):
        return cv2.dnn.blobFromImage(cv2.resize(self.image, blob_size), 0.007843, blob_size, 127.5)
