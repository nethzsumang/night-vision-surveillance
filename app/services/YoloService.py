import cv2
import numpy as np


class YoloService:
    def __init__(self, config):
        self.config = config
        self.labels = self.load_label()
        [self.ln, self.net] = self.load_network()
        np.random.seed(42)

    def load_label(self):
        try:
            return open(self.config["paths"]["label"]).read().strip().split("\n")
        except KeyError:
            print("Label path not found in config.")
            return None

    def load_network(self):
        try:
            net = cv2.dnn.readNetFromDarknet(self.config["paths"]["cfg"], self.config["paths"]["weights"])
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            return [ln, net]
        except KeyError:
            print("Config and/or weights path not found in config")

    def forward_pass(self, frame, scale=1/255, size=(416, 416)):
        blob = cv2.dnn.blobFromImage(frame, scale, size)
        self.net.setInput(blob)
        return self.net.forward(self.ln)

    def process_output(self, frame, layer_outputs, width, height):
        boxes = []
        confidences = []
        classIDs = []

        coordinates = []
        colors = []
        texts = []
        label_names = []
        accuracies = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.config["settings"]["confidence"] and classID in self.config["settings"]["necessary_classes"]:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.config["settings"]["confidence"], self.config["settings"]["threshold"])
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [0, 0, 255]
                text = "{}: {:.4f}".format(self.labels[classIDs[i]], confidences[i])
                accuracy = confidences[i] * 100

                coordinates.append([x, y, w, h])
                colors.append(color)
                texts.append(text)
                accuracies.append(accuracy)
                label_names.append(self.labels[classIDs[i]])

        return [coordinates, colors, texts, accuracies, label_names]
