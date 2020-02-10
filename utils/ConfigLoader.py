class ConfigLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_config(path):
        import json
        import os

        data = None
        cwd = os.getcwd()
        with open(cwd + os.sep + path, 'r') as file:
            data = json.load(file)
        return data
