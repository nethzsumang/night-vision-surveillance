import cv2


class Image:
    """
        Image class
    """

    # width of the image
    width = 0

    # height of the image
    height = 0

    # image itself
    image = None

    def __init__(self, path):
        """
            Constructor.

            :param path:
        """
        self.image = cv2.imread(path)
        [self.height, self.width, _] = self.image.shape

    def get_input_blob(self, blob_size):
        """
            Get input blob to be used in the network

            :param blob_size:
            :return:
        """
        return cv2.dnn.blobFromImage(cv2.resize(self.image, blob_size), 0.007843, blob_size, 127.5)
