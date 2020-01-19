from app.models.Thread import Thread
from app.services.FPSService import FPSService
from app.services.VideoStreamService import VideoStreamService
from app.services.YoloService import YoloService
from app.services.ImageProcessingService import ImageProcessingService
from app.services.VideoWriterService import VideoWriterService
import cv2
from datetime import date


def process(config):
    filename = config["storage"]["video_output_dir"] + "surveillance_" + date.today().strftime("%Y-%m-%d_%H:%M:%S") + ".mp4"

    fps = FPSService()
    video_stream = VideoStreamService()
    yolo_service = YoloService(config)
    video_writer = VideoWriterService(filename)

    video_stream.start()
    fps.start()

    while True:
        [frame, h, w] = video_stream.get_frame()
        layer_output = yolo_service.forward_pass(
            frame,
            scale=float(config["settings"]["scale"]),
            size=tuple(config["settings"]["frame_size"])
        )
        [coordinates, colors, texts] = yolo_service.process_output(frame, layer_output, w, h)

        for coordinate, color, text in zip(coordinates, colors, texts):
            ImageProcessingService.draw_rectangle(frame, coordinate, color)
            ImageProcessingService.put_text(frame, text, (coordinate[0], coordinate[1] - 5), color)

        # pass frame to save
        video_writer.write(frame)
        ImageProcessingService.show_image(frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    video_writer.release()
    video_stream.stop()
    fps.stop()
    cv2.destroyAllWindows()
