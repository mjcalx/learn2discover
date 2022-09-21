import numpy as np
from data.schema import Schema
from data.loader import Loader
from data.data_classes import DataAttributes
from data.data_classes import Label
from configs.config_manager import ConfigManager
from loggers.logger_factory import LoggerFactory
import pandas as pd
# ! WIP
class DatasetManager:
    def __init__(self, random_state: int=42):
        """_summary_
        Args:
            random_state (int, optional): random seed. Defaults to 42.
        """
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self.random = np.random.RandomState(random_state)
        self.loader = Loader()
        self.schema, self.data = self.loader.load_data()

        self.attributes = None
        self.X = None
        self.Y = None
        self.outcomes = pd.Series([None]*len(self.data))
        self.fairness_labels = pd.Series([None]*len(self.data))
        self.labelling_scheme = None

    def parse_data_instances(self, attributes: DataAttributes):
        self.attributes = attributes
        self.X = self.data[self.attributes.inputs]
        self.Y = self.data[self.attributes.outputs]

    def label_instance(self, idx: int, label: Label):
        self.fairness_labels[idx] = label

    def get_instance(self, idx: int) -> pd.Series:
        combined = pd.concat([self.X.loc[idx], self.Y.loc[idx]])
        return combined