from app.models.Thread import Thread
from app.services.FPSService import FPSService
from app.services.VideoStreamService import VideoStreamService
from app.services.YoloService import YoloService
from app.services.ImageProcessingService import ImageProcessingService
from app.services.VideoWriterService import VideoWriterService
import cv2
import datetime
import time


class App:
    def __init__(self, config):
        self.config = config
        self.fps_service = FPSService()
        self.video_stream_service = VideoStreamService()
        self.yolo_service = YoloService(config)
        self.video_writer_service = VideoWriterService("")

    def process(self):
        try:
            self.main_process()
        except Exception as e:
            print(str(e))
            self.fps_service.stop()
            self.video_stream_service.stop()
            self.video_writer_service.release()

    def main_process(self):
        filename = App.get_filename()
        frame_arr = []

        self.video_stream_service.start()
        self.fps_service.start()

        # for the record of video
        video_length = int(self.config["storage"]["video_length"])

        skip_frames = int(self.config["settings"]["skip_frames"])
        time_start = time.time()

        to_skip = 0

        while True:
            [frame, h, w] = self.video_stream_service.get_frame()
            if to_skip == 0:
                layer_output = self.yolo_service.forward_pass(
                    frame,
                    scale=float(self.config["settings"]["scale"]),
                    size=tuple(self.config["settings"]["frame_size"])
                )
                [coordinates, colors, texts] = self.yolo_service.process_output(frame, layer_output, w, h)

                for coordinate, color, text in zip(coordinates, colors, texts):
                    ImageProcessingService.draw_rectangle(frame, coordinate, color)
                    ImageProcessingService.put_text(frame, text, (coordinate[0], coordinate[1] - 5), color)
                to_skip = skip_frames
            else:
                to_skip = to_skip - 1

            time_diff = time.time() - time_start
            if time_diff >= video_length:
                filename = App.get_filename()
                frame_dim = frame.shape
                self.video_writer_service = VideoWriterService(filename, dimensions=(frame_dim[1], frame_dim[0]))
                thread = Thread(
                    App.video_writer_fun,
                    [
                        self.video_writer_service,
                        frame_arr,
                        self.config
                    ],
                    1,
                    "video_writer",
                    delay=0
                )
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
            self.fps_service.update()

        self.video_writer_service.release()
        self.video_stream_service.stop()
        self.fps_service.stop()
        cv2.destroyAllWindows()

    @staticmethod
    def video_writer_fun(args):
        [writer, frame_arr, config] = args
        for frame in frame_arr:
            for x in range(0, int(config["settings"]["repeat_frames"])):
                writer.write(frame)
        writer.release()

    @staticmethod
    def get_filename():
        return "surveillance_" + \
               datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
