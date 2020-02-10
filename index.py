from utils.ConfigLoader import ConfigLoader
from app.app import process
from var_dump import var_dump


config_data = ConfigLoader.load_config("config/app.json")
process(config_data)
