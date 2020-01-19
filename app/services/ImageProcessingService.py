import cv2


class ImageProcessingService:
    def __init__(self):
        pass

    @staticmethod
    def draw_rectangle(image, coordinates, color):
        """
            coordinates is [x, y, w, h]
            :param image:
            :param coordinates:
            :param color:
            :return:
        """
        bounds = {(coordinates[0] + coordinates[2]), (coordinates[1] + coordinates[3])}
        cv2.rectangle(image, set[coordinates[0:1]], bounds, color, 2)

    @staticmethod
    def put_text(image, text, position, color):
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
