import cv2
from constants import PROTOTEXT, MODEL


def read_network():
    return cv2.dnn.readNetFromCaffe(PROTOTEXT, MODEL)
