from app.models.Thread import Thread
from app.services.FPSService import FPSService
from app.services.VideoStreamService import VideoStreamService
from app.services.YoloService import YoloService
from app.services.ImageProcessingService import ImageProcessingService


def process(config):
    fps = FPSService()
    video_stream = VideoStreamService()
    yolo_service = YoloService(config)

    video_stream.start()
    fps.start()

    while True:
        [frame, h, w] = video_stream.get_frame()
        layer_output = yolo_service.forward_pass(frame)
        [coordinates, colors, texts] = yolo_service.process_output(frame, layer_output)

        for coordinate, color, text in zip(coordinates, colors, texts):
            ImageProcessingService.draw_rectangle(frame, coordinate, color)
            ImageProcessingService.put_text(frame, text, (coordinate[0], coordinate[1] - 5), color)

        # pass frame to save
