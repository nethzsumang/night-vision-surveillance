from imutils.video import FPS


class FPSService:
    def __init__(self):
        self.fps = FPS()

    def start(self):
        self.fps.start()

    def update(self):
        self.fps.update()

    def stop(self):
        self.fps.stop()
