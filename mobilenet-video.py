import cv2

from constants.mobilenet import CLASSES, IMPORTANT_CLASSES, IMAGE_SIZE, MODEL, PROTOTEXT, COLORS
from app.video import Video
from app.image import Image
from app.mobilenet import Network

network = Network(MODEL, PROTOTEXT, confidence=0.8, important_classes=IMPORTANT_CLASSES)
video_obj = Video()
video_obj.start()

while True:
    frame = video_obj.video_stream.read()
    (h, w) = frame.shape[:2]
    image_obj = Image(image=frame)
    blob = image_obj.get_input_blob(IMAGE_SIZE)
    network.set_input(blob)
    detections = network.run()

    for detection in detections:
        confidence = detection.confidence
        idx = detection.classification
        [startX, startY, endX, endY] = detection.get_real_coordinates(image_obj.width, image_obj.height)

        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        cv2.rectangle(image_obj.image, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image_obj.image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    video_obj.fps.update()

video_obj.fps.stop()
print("[INFO] elapsed time: {:.2f}".format(video_obj.fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(video_obj.fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
video_obj.video_stream.stop()
