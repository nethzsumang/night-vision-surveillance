from app.models.Thread import Thread
from app.services.FPSService import FPSService
from app.services.VideoStreamService import VideoStreamService
from app.services.YoloService import YoloService
from app.services.ImageProcessingService import ImageProcessingService
from app.services.VideoWriterService import VideoWriterService
import cv2
import datetime
import time


def process(config):
    filename = get_filename(config)
    frame_arr = []

    fps = FPSService()
    video_stream = VideoStreamService()
    yolo_service = YoloService(config)

    video_stream.start()
    fps.start()

    # for the record of video
    video_length = int(config["storage"]["video_length"])
    time_start = time.time()

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

        time_diff = time.time() - time_start
        if time_diff >= video_length:
            filename = get_filename(config)
            frame_dim = frame.shape
            video_writer = VideoWriterService(filename, dimensions=(frame_dim[1], frame_dim[0]))
            thread = Thread(video_writer_fun, [video_writer, frame_arr], 1, "video_writer", delay=0)
            thread.start()
            time_start = time.time()
        else:
            frame_arr.append(frame)

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


def video_writer_fun(args):
    [writer, frame_arr] = args
    for frame in frame_arr:
        writer.write(frame)
    writer.release()


def get_filename(config):
    return "surveillance_" + \
           datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"