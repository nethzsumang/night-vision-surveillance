import cv2


class Network:
    labels = None
    network = None
    layer_names = None

    def __init__(self, labels, weights_path, cfgs_path):
        self.labels = labels
        self.network = cv2.dnn.readNetFromDarknet(cfgs_path, weights_path)
        self.layer_names = self.network.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.network.getUnconnectedOutLayers()]

    def set_input(self, input_data):
        self.network.setInput(input_data)

    def run(self):
        return self.network.forward()
