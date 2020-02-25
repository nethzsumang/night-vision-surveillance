import cv2


class VideoWriterService:
    def __init__(self, filename, dimensions=(480, 640)):
        self.filename = filename
        self.writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 20, dimensions)
        print("Writer instantiated with dimensions (" + str(dimensions[0]) + ", " + str(dimensions[1]) + ")")

    def write(self, frame):
        self.writer.write(frame)
        print("Video written at " + self.filename)

    def release(self):
        self.writer.release()
        print("Writer released.")
