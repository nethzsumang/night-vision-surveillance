import threading
import time


class Thread(threading.Thread):
    def __init__(self, process, args, threadID, name, delay=1):
        threading.Thread.__init__(self)
        self.daemon = True
        self.threadID = threadID
        self.name = name
        self.delay = delay
        self.process = process
        self.args = args

    def run(self):
        self.process(self.args)
        time.sleep(self.delay)
