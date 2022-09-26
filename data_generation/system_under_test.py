import pandas as pd
from loggers.logger_factory import LoggerFactory

class SystemUnderTest:
    def __init__(self):
        self.logger = LoggerFactory.get_logger(self._type())
    
    def _type(self):
        return self.__class__.__name__

    def evaluate_outcomes(self, outputs: pd.DataFrame) -> pd.Series:
        pass