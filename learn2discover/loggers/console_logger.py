from loggers.logger import Logger
from datetime import datetime


class ConsoleLogger(Logger):

    def __init__(self, class_name, log_level):
        super().__init__(class_name, log_level)

    def debug(self, message):
        if self.log_level == 'debug':
            current_datetime = datetime.now()
            self._msg_format('DEBUG',message)

    def info(self, message):
        if self.log_level == 'info':
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