import os
import numpy as np


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
PROTOTEXT = os.getcwd() + os.sep + "resources" + os.sep + "MobileNetSSD_deploy.prototxt.txt"
MODEL = os.getcwd() + os.sep + "resources" + os.sep + "MobileNetSSD_deploy.caffemodel"
IMAGE = os.getcwd() + os.sep + "resources" + os.sep + "images" + os.sep + "bus_and_car.jpg"

CONFIDENCE = 0.5

IMAGE_SIZE = (300, 300)

OBJECT = "image"
