from imutils.video import FPS


class FPSService:
    def __init__(self):
        self.fps = FPS()

    def start(self):
        self.fps.start()
