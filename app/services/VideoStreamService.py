from imutils.video import VideoStream
import time


class VideoStreamService:
    def __init__(self, src=0, start_delay=2):
        self.stream = VideoStream(src)
        time.sleep(start_delay)

    def start(self):
        self.stream = self.stream.start()

    def get_frame(self):
        frame = self.stream.read()
        (H, W) = frame.shape[:2]
        return [frame, H, W]

    def stop(self):
        self.stream.stop()
