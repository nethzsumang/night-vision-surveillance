from imutils.video import FPS, VideoStream
import imutils
import time


class Video:
    video_stream = None
    fps = None

    def __init__(self):
        pass

    def start(self):
        self.video_stream = VideoStream(src=0).start()
        time.sleep(2)
        self.fps = FPS().start()

    def read_frame(self):
        frame = self.video_stream.read()
        return imutils.resize(frame, width=400)


