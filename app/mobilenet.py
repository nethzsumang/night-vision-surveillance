import cv2
import numpy as np
from constants import PROTOTEXT, MODEL, CONFIDENCE


class Network:
    input = None
    network = None
    detections = None

    def __init__(self, model_path, prototext_path):
        self.network = cv2.dnn.readNetFromCaffe(PROTOTEXT, MODEL)

    def set_input(self, input_image):
        self.network.setInput(input_image)

    def run(self):
        detections = np.array([Detection(detection) for detection in self.network.forward()[0, 0] if detection[2] > CONFIDENCE])
        return detections


class Detection:
    confidence = 0.
    coordinates = []
    classification = 0

    def __init__(self, arr):
        self.confidence = arr[2]
        self.coordinates = arr[3:7]
        self.classification = arr[1]
