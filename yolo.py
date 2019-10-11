import cv2
import numpy as np

from app.image import Image
from app.yolo import Network
from app.video import Video
from constants.yolo import LABELS, COLORS, WEIGHTS_PATH, CFGS_PATH, CONFIDENCE, THRESHOLD


network_obj = Network(LABELS, WEIGHTS_PATH, CFGS_PATH)
video_obj = Video()
video_obj.start()

while True:
    frame = video_obj.video_stream.read()
    (h, w) = frame.shape[:2]
    image_obj = Image(image=frame)
    blob = image_obj.get_input_blob(416, swapRB=True)
    network_obj.set_input(blob)
    layer_outputs = network_obj.run()

    boxes = []
    confidences = []
    classIDs = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                           confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    video_obj.fps.update()

video_obj.video_stream.release()
