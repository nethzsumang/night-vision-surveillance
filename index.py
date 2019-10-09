import numpy as np
import cv2

from constants import CLASSES, IMAGE_PATH, IMAGE_SIZE, MODEL, PROTOTEXT, COLORS
from app.image import Image
from app.mobilenet import Network

network = Network(MODEL, PROTOTEXT)
image_obj = Image(IMAGE_PATH)
blob = image_obj.get_input_blob(IMAGE_SIZE)
network.set_input(blob)
detections = network.run()

for detection in detections:
    confidence = detection.confidence
    idx = int(detection.classification)
    [startX, startY, endX, endY] = detection.get_real_coordinates(image_obj.width, image_obj.height)

    # display the prediction
    cv2.rectangle(image_obj.image, (startX, startY), (endX, endY), COLORS[idx], 2)
    y = startY - 15 if startY - 15 > 15 else startY + 15
    cv2.putText(image_obj.image, "{}: {:.2f}%".format(CLASSES[idx], confidence * 100), (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

cv2.imshow("Output", image_obj.image)
cv2.waitKey(0)
