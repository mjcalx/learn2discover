from abc import abstractmethod

import pandas as pd
from data.data_attributes import DataAttributes
from loggers.logger_factory import LoggerFactory


class AbstractSystemUnderTest:
    def __init__(self, input_attributes, output_attributes, oracle_args):
        self.logger = LoggerFactory.get_logger(self._type())
        self.attributes = DataAttributes(input_attributes, output_attributes)
        self.oracle_args = oracle_args
    
    def _type(self):
        return self.__class__.__name__

    @abstractmethod
    def evaluate_outcomes(self, outputs: pd.DataFrame) -> pd.Series:
        pass
