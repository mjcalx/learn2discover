from typing import List, Callable, TypeVar
import pandas as pd
from data.data_classes import Outcome
from loggers.logger_factory import LoggerFactory


class MockOracle:
    def __init__(self):
        self.logger = LoggerFactory.get_logger(self._type())

    def _type(self):
        return self.__class__.__name__

    def set_labels(self, *args, **kwargs) -> pd.Series:
        pass
