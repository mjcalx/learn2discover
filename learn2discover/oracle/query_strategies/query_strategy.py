import pandas as pd
import torch.nn as nn

from loggers.logger_factory import LoggerFactory
from data.dataset_manager import DatasetManager

class QueryStrategy:

    def __init__(self):
        self.logger = LoggerFactory.get_logger(self._type())
        self.dataset_manager = DatasetManager.get_instance()

        assert isinstance(self.dataset_manager, DatasetManager)

    @property
    def name(self):
        return 'Generic Query Strategy'
    
    def _type(self):
        return self.__class__.__name__
    
    def query(self, classifier: nn.Module, *args, **kwargs) -> pd.DataFrame:
        pass
    
