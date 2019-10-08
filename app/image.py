import cv2

from constants import IMAGE, IMAGE_SIZE


def get_input_blob():
    image = cv2.imread(IMAGE)
    (h, w) = image.shape[:2]
    return [
        image,
        cv2.dnn.blobFromImage(cv2.resize(image, IMAGE_SIZE), 0.007843, IMAGE_SIZE, 127.5),
        h,
        w
    ]
