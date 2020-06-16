import pygame


class AlertService:
    def __init__(self, config):
        self.config = config
        pygame.mixer.init()

    def alert(self):
        if self.config["audio"]["audio_alert"] == 1:
            pygame.mixer.stop()
            pygame.mixer.music.load("Alarm.mp3")
            pygame.mixer.music.play()
