from utils.ConfigLoader import ConfigLoader
from app.app import App
import sys


config_data = ConfigLoader.load_config("config/app.json")
App(config_data).process()
sys.exit(0)
