import cv2

from constants import IMAGE_SIZE


def read_image(path):
    return cv2.imread(path)


def get_input_blob(image):
    (h, w) = image.shape[:2]
    return [
        cv2.dnn.blobFromImage(cv2.resize(image, IMAGE_SIZE), 0.007843, IMAGE_SIZE, 127.5),
        h,
        w
    ]
