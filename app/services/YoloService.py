import cv2
import numpy as np


class YoloService:
    def __init__(self, config):
        self.config = config
        self.labels = self.load_label()
        [self.ln, self.net] = self.load_network()

    def load_label(self):
        try:
            return open(self.config["paths"]["label"]).read().strip().split("\n")
        except KeyError:
            print("Label path not found in config.")
            return None

    def load_network(self):
        try:
            net = cv2.dnn.readNetFromDarknet(self.config["paths"]["config"], self.config["paths"]["weights"])
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            return [ln, net]
        except KeyError:
            print("Config and/or weights path not found in config")

    def forward_pass(self, frame, scale=1/255, size=(416, 416)):
        blob = cv2.dnn.blobFromImage(frame, scale, size)
        self.net.setInput(blob)
        return self.net.forward(self.ln)

    def process_output(self, layer_outputs, width, height):
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.config["settings"]["confidence"] and classID in self.config["settings"]["necessary_classes"]:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, width, height) = box.astype("int")

