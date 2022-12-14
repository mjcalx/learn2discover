class Logger:

    def __init__(self, class_name, log_level, verbosity):
        self.class_name = class_name
        self.log_level = log_level
        self.verbosity = verbosity

    def debug(self, message):
        pass

    def info(self, message):
        pass

    def warn(self, message):
        pass

    def error(self, message):
        pass
