from app.models.Thread import Thread
from app.services.FPSService import FPSService
from app.services.VideoStreamService import VideoStreamService
from app.services.YoloService import YoloService


def process(config):
    fps = FPSService()
    video_stream = VideoStreamService()
    yolo_service = YoloService(config)

    video_stream.start()
    fps.start()

    while True:
        [frame, h, w] = video_stream.get_frame()
        layerOutput = yolo_service.forward_pass(frame)
