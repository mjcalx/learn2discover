import numpy as np
from datetime import datetime
from configs.config_manager import ConfigManager


class LoggingUtils:
    instance = None

    def __init__(self):
        self.primary_logger_type = ConfigManager.get_instance().primary_logger_type
        self.log_level = ConfigManager.get_instance().log_level
        self.verbosity = ConfigManager.get_instance().verbosity

    @staticmethod
    def get_instance():
        if LoggingUtils.instance is None:
            LoggingUtils.instance = LoggingUtils()

        return LoggingUtils.instance

    def debug(self, message):
        if self.log_level == 'debug':
            print(message)

    def info(self, message):
        if self.log_level == 'info':
            print(message)

    def warn(self, message):
        print(message)

    def error(self, message):
        current_datetime = datetime.now()
        print('[ERROR][' + str(current_datetime) + ']' + message)

    def get_primary_logger_type(self):
        return self.primary_logger_type

    def get_log_level(self):
        return self.log_level

    def get_verbosity(self):
        return self.verbosity
