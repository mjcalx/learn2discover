from loggers.console_logger import ConsoleLogger
from utils.logging_utils import LoggingUtils


class LoggerFactory:

    @staticmethod
    def get_logger(class_name):
        primary_logger_type = LoggingUtils.get_instance().get_primary_logger_type()
        if primary_logger_type == "console":
            log_level = LoggingUtils.get_instance().get_log_level()
            return ConsoleLogger(class_name, log_level)
        else:
            LoggingUtils.get_instance().error('[%s] Logger type %s is not supported!' %
                                              (__class__.__name__, primary_logger_type))
