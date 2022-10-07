from loggers.logger_factory import LoggerFactory

class StoppingCriterion:
    def __init__(self):
        self.logger = LoggerFactory.get_logger(self._type())
    
    def _type(self):
        return self.__class__.__name__

    def __call__(self) -> bool:
        pass

    def report(self) -> None:
        pass
        