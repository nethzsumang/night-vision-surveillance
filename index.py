import numpy as np
import cv2

from constants import CLASSES, IMAGE, MODEL, PROTOTEXT, COLORS
from app.image import read_image, get_input_blob
from app.mobilenet import Network

network = Network(MODEL, PROTOTEXT)
image = read_image(IMAGE)
[blob, h, w] = get_input_blob(image)
network.set_input(blob)
detections = network.run()

for detection in detections:
    confidence = detection.confidence
    idx = int(detection.classification)
    box = detection.coordinates * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    # display the prediction
    cv2.rectangle(image, (startX, startY), (endX, endY),
                  COLORS[idx], 2)
    y = startY - 15 if startY - 15 > 15 else startY + 15
    cv2.putText(image, "{}: {:.2f}%".format(CLASSES[idx], confidence * 100), (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
