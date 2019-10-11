import numpy as np
import os


LABELS_PATH = os.getcwd() + os.sep + "resources" + os.sep + "yolo" + os.sep + "coco.names"
WEIGHTS_PATH = os.getcwd() + os.sep + "resources" + os.sep + "yolo" + os.sep + "yolov3.weights"
CFGS_PATH = os.getcwd() + os.sep + "resources" + os.sep + "yolo" + os.sep + "yolov3.cfg"
LABELS = open(LABELS_PATH).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

CONFIDENCE = 0.5
THRESHOLD = 0.3