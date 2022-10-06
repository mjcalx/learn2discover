from loggers.logger import Logger
from utils.logging_utils import Verbosity
from datetime import datetime

class ConsoleLogger(Logger):

    def __init__(self, class_name, log_level, verbosity_int):
        super().__init__(class_name, log_level, verbosity_int)

    def debug(self, message, verbosity=Verbosity.BASE):
        if self.log_level == 'debug' and verbosity.value <= self.verbosity:
            current_datetime = datetime.now()
            self._msg_format('DEBUG',message)

    def info(self, message, verbosity=Verbosity.BASE):
        if self.log_level in ['info', 'debug'] and verbosity.value <= self.verbosity:
            current_datetime = datetime.now()
            self._msg_format('INFO',message)

    def warn(self, message):
        current_datetime = datetime.now()
        self._msg_format('WARN',message)

    def error(self, message):
        current_datetime = datetime.now()
        self._msg_format('ERROR',message)
    
    def _msg_format(self, type_str, msg):
        print(f'[{type_str}][{str(datetime.now())}][{self.class_name}] {msg}')