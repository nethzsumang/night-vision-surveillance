from app.models.Thread import Thread
from app.services.FPSService import FPSService
from app.services.VideoStreamService import VideoStreamService
from app.services.YoloService import YoloService
from app.services.ImageProcessingService import ImageProcessingService
from app.services.VideoWriterService import VideoWriterService
from app.services.AlertService import AlertService
import cv2
import datetime
import time
import sys


class App:
    def __init__(self, config):
        self.config = config
        self.fps_service = FPSService()
        self.video_stream_service = VideoStreamService(src=int(config["stream_settings"]["src"]))
        self.yolo_service = YoloService(config)
        self.video_writer_service = VideoWriterService("")
        self.alert_service = AlertService(config)
        self.frame_arr = []

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
                [coordinates, colors, texts, accuracies, label_names] = self.yolo_service.process_output(
                    frame,
                    layer_output,
                    w,
                    h
                )
                copy = frame.copy()

                for coordinate, color, text, accuracy, lbl in zip(coordinates, colors, texts, accuracies, label_names):
                    print(lbl + " detected on " + str(coordinate) + ". (Accuracy: " + str(int(accuracy)) + "%)")
                    ImageProcessingService.draw_rectangle(copy, coordinate, color)
                    ImageProcessingService.put_text(copy, text, (coordinate[0], coordinate[1] - 5), color)
                    ImageProcessingService.save_image(copy, App.get_filename(".png"))
                    self.alert_service.alert()
                to_skip = skip_frames
            else:
                to_skip = to_skip - 1

            time_diff = time.time() - time_start
            if time_diff >= video_length:
                filename = App.get_filename()
                frame_dim = copy.shape
                self.video_writer_service = VideoWriterService(filename, dimensions=(frame_dim[1], frame_dim[0]))
                video_writer_thread = Thread(self.video_writer_fun, [], 1, "video_writer", delay=0)
                image_saver_thread = Thread(self.image_save_fun, copy, 2, "image_saver", delay=0)
                video_writer_thread.start()
                image_saver_thread.start()
                time_start = time.time()
            else:
                self.frame_arr.append(copy)

            ImageProcessingService.show_image(copy)
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

    def video_writer_fun(self, args):
        for frame in self.frame_arr:
            # for x in range(0, int(self.config["settings"]["repeat_frames"])):
            self.video_writer_service.write(frame)
        self.frame_arr.clear()
        self.video_writer_service.release()
        # exit thread
        sys.exit()

    def image_save_fun(self, frame):
        ImageProcessingService.save_image(frame, self.get_filename(".png"))
        # exit thread
        sys.exit()

    @staticmethod
    def get_filename(ext=".mp4"):
        return "surveillance_" + \
               datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ext
