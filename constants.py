import os


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

PROTOTEXT = os.getcwd() + os.sep + "resources" + os.sep + "MobileNetSSD_deploy.prototxt.txt"
MODEL = os.getcwd() + os.sep + "resources" + os.sep + "MobileNetSSD_deploy.caffemodel"
IMAGE = os.getcwd() + os.sep + "resources" + os.sep + "images" + os.sep + "example_02.jpg"

CONFIDENCE = 0.2

IMAGE_SIZE = (300, 300)