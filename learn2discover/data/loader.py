from data.schema import Schema
import os
from loggers.logger_factory import LoggerFactory

class Loader:
    def __init__(self):
        pass

    @staticmethod
    def get_yaml_configs(workspace_dir=os.getcwd(), config_file=None):
        yaml_path = os.path.join(workspace_dir), config_file
        configs = {}
        if os.path.exists(yaml_path):
            stream = open(yaml_path, "r")
            configs = yaml.full_load(stream)
        return configs

