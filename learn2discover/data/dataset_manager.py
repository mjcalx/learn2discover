from __future__ import annotations
import numpy as np
from data.schema import Schema
from data.loader import Loader
from data.data_classes import DataAttributes, Label
from configs.config_manager import ConfigManager
from loggers.logger_factory import LoggerFactory
import pandas as pd

class DatasetManager:
    instance = None

    def __init__(self, random_state: int=42, attributes: DataAttributes=None):
        """_summary_
        Args:
            random_state (int, optional): random seed. Defaults to 42.
        """
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self.random = np.random.RandomState(random_state)
        self.loader = Loader()
        self.schema, self.data = self.loader.load_data()

        self.attributes = attributes
        self.X = None
        self.Y = None
        if self.attributes is not None:
            self.parse_data_instances()
        
        self.outcomes = pd.Series([None]*len(self.data))
        self.fairness_labels = pd.Series([None]*len(self.data))
        DatasetManager.instance = self
    
    @staticmethod
    def get_instance() -> DatasetManager:
        if DatasetManager.instance is None:
            DatasetManager.instance = DatasetManager()
        return DatasetManager.instance

    def parse_data_instances(self, attributes: DataAttributes=None) -> None:
        if self.attributes is None and attributes is None:
            raise ValueError("Unset DataManager.attributes")
        if attributes is not None:
            self.attributes = attributes
        self.X = self.data[self.attributes.inputs]
        self.Y = self.data[self.attributes.outputs]
