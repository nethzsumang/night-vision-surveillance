import cv2


class VideoWriterService:
    def __init__(self, filename):
        self.writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"MP4V"), 20, (416, 416))

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
