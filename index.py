import numpy as np
import cv2

from constants import CLASSES, CONFIDENCE
from app.image import get_input_blob
from app.mobilenet import read_network

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

network = read_network()
[image, blob, h, w] = get_input_blob()
network.setInput(blob)
detections = network.forward()

for i in np.arange(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > CONFIDENCE:
        # extract the index of the class label from the `detections`,
        # then compute the (x, y)-coordinates of the bounding box for
        # the object
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # display the prediction
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print("[INFO] {}".format(label))
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
