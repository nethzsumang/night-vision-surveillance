import cv2
import numpy as np


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

    # confidence level threshold
    confidence_threshold = 0.

    def __init__(self, model_path, prototext_path, confidence=0.2):
        """
            Constructor.

            :param model_path:
            :param prototext_path:
        """
        self.network = cv2.dnn.readNetFromCaffe(prototext_path, model_path)
        self.confidence_threshold = confidence

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
            [
                Detection(detection)
                for detection in self.network.forward()[0, 0] if detection[2] > self.confidence_threshold
            ]
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
        self.classification = int(arr[1])

    def get_real_coordinates(self, width, height):
        """
            Gets the real coordinates based on the height
            and width of the real image

            :param width:
            :param height:
            :return:
        """
        return (self.coordinates * np.array([width, height, width, height])).astype("int")
