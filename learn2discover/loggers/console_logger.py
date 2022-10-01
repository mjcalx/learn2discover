from loggers.logger import Logger
from datetime import datetime


class ConsoleLogger(Logger):

    def __init__(self, class_name, log_level, verbosity):
        super().__init__(class_name, log_level, verbosity)

    def debug(self, message, verbosity=0):
        if self.log_level == 'debug' and verbosity <= self.verbosity:
            current_datetime = datetime.now()
            self._msg_format('DEBUG',message)

    def info(self, message, verbosity=0):
        if self.log_level == 'info' and verbosity <= self.verbosity:
            current_datetime = datetime.now()
            self._msg_format('INFO',message)

    def warn(self, message, verbosity=0):
        current_datetime = datetime.now()
        self._msg_format('WARN',message)

    def error(self, message, verbosity=0):
        current_datetime = datetime.now()
        self._msg_format('ERROR',message)
    
    def _msg_format(self, type_str, msg):
        print(f'[{type_str}][{str(datetime.now())}][{self.class_name}] {msg}')