from picamera import PiCamera
import numpy as np
import time
import cv2
import os


LABELS_PATH = "resources" + os.sep + "yolo" + os.sep + "coco.names"
WEIGTHS_PATH = "resources" + os.sep + "yolo" + os.sep + "yolov3-tiny.weights"
CFG_PATH = "resources" + os.sep + "yolo" + os.sep + "yolov3.cfg"
LABELS = open(LABELS_PATH).read().strip().split("\n")
NECESSARY_CLASSES = [0, 16, 15]
CONFIDENCE = 0.3
THRESHOLD = 0.3

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGTHS_PATH)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
camera = PiCamera()
camera.resolution = (416, 416)
