from loggers.logger_factory import LoggerFactory
from data.dataset_manager import DatasetManager
import torch.nn as nn

class QueryStrategy:

    def __init__(self):
        self.logger = LoggerFactory.get_logger(self._type())
        self.dataset_manager = DatasetManager.get_instance()

        assert isinstance(self.dataset_manager, DatasetManager)

    
    def _type(self):
        return self.__class__.__name__

    
    def __name__(self):
        return 'Generic Query Strategy'
    
    def query(self, classifier: nn.Module) -> None:
        pass
    
