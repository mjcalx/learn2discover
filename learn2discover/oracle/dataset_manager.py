import numpy as np
from data.schema import Schema
from data.loader import Loader
from configs.config_manager import ConfigManager
from loggers.logger_factory import LoggerFactory
# ! WIP
class DatasetManager:
    def __init__(self, random_state: int=42):
        """_summary_
        Args:
            random_state (int, optional): random seed. Defaults to 42.
        """
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self.random = np.random.RandomState(random_state)
        self.schema = None
        self.loader = Loader()
        self.data   = self.loader.load_data()
        self.already_labelled = {}  # tracking what is already labelled
        self.feature_idex = {} # feature mapping for one-hot encoding
