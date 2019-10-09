import cv2
import numpy as np
from constants import PROTOTEXT, MODEL, CONFIDENCE


class Network:
    """
        MobileNet Network class
    """

    # input for the network
    input = None

    # network object
    network = None

    # result of the network
    detections = None

    def __init__(self, model_path, prototext_path):
        """
            Constructor.

            :param model_path:
            :param prototext_path:
        """
        self.network = cv2.dnn.readNetFromCaffe(PROTOTEXT, MODEL)

    def set_input(self, input_image):
        """
            Sets the input in the network.

            :param input_image:
            :return:
        """
        self.network.setInput(input_image)

    def run(self):
        """
            Runs the network

            :return: np.array
        """
        detections = np.array(
            [Detection(detection) for detection in self.network.forward()[0, 0] if detection[2] > CONFIDENCE]
        )
        return detections


class Detection:
    """
        Detection class
    """

    # Confidence level of the detection
    confidence = 0.

    # coordinates of the detection
    coordinates = []

    # classification of the detection
    classification = 0

    def __init__(self, arr):
        """
            Constructor.

            :param arr:
        """
        self.confidence = arr[2]
        self.coordinates = arr[3:7]
        self.classification = arr[1]

    def get_real_coordinates(self, width, height):
        return (self.coordinates * np.array([width, height, width, height])).astype("int")