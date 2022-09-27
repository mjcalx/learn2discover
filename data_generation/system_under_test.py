import pandas as pd
from loggers.logger_factory import LoggerFactory
from data.data_classes import DataAttributes

class SystemUnderTest:
    def __init__(self, input_attributes, output_attributes):
        self.logger = LoggerFactory.get_logger(self._type())
        self.attributes = DataAttributes(input_attributes, output_attributes)
    
    def _type(self):
        return self.__class__.__name__

    def evaluate_outcomes(self, outputs: pd.DataFrame) -> pd.Series:
        pass