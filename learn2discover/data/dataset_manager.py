from __future__ import annotations
import numpy as np
from data.schema import Schema
from data.loader import Loader
from data.data_classes import DataAttributes, Label, Outcome, ParamType
from configs.config_manager import ConfigManager
from loggers.logger_factory import LoggerFactory
import pandas as pd

class DatasetManager:
    instance = None

    def __init__(self, attributes: DataAttributes=None, random_state: int=42):
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
        self.dataset = None
        
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

    def set_outcomes(self, outcomes: pd.Series) -> None:
        m = '"outcomes" must be of type Series[Outcome]'
        assert all(outcomes.apply(lambda x : isinstance(x, Outcome))), m
        n = '"outcomes" must have same length as data'
        assert len(self.outcomes) == len(self.X), n

        self.outcomes = outcomes
        self.outcomes.rename(ParamType.OUTCOME.value, inplace=True)

    def set_fairness_labels(self, fairness_labels: pd.Series) -> None:
        m = '"fairness_labels" must be of type Series[Label]'
        assert all(fairness_labels.apply(lambda x : isinstance(x, Label))), m
        n = '"fairness_labels" must have same length as data'
        assert len(self.fairness_labels) == len(self.X), n

        self.fairness_labels = fairness_labels
        self.fairness_labels.rename(ParamType.FAIRNESS.value, inplace=True)

    def format_dataset(self) -> pd.DataFrame:
        """
        Combine and reindex the dataset with SUT outcomes and fairness oracle labels.

        Returns:
            pd.DataFrame: _description_
        """
        if self.dataset is not None:
            return self.dataset
        m = '"All data must be instantiated before calling "format_dataset()"'
        data = [self.X, self.Y, self.outcomes, self.fairness_labels]
        assert all([d is not None for d in data]), m

        df = pd.concat([self.data, self.outcomes, self.fairness_labels], axis=1)
        levels = [
                *[ParamType.INPUTS.value]*len(self.attributes.inputs), 
                *[ParamType.OUTPUTS.value]*len(self.attributes.outputs), 
                ParamType.OUTCOME.value, ParamType.FAIRNESS.value]
        codes = [*self.attributes.inputs, *self.attributes.outputs, 
                self.outcomes.name, self.fairness_labels.name]
        assert len(levels) == len(codes)

        tup = list(zip(*[levels, codes]))
        cols = pd.MultiIndex.from_tuples(tup, names=('param_type', 'param'))
        df.columns = cols
        self.dataset = df
        return self.dataset

    def save_dataset(self):
        if self.dataset is None:
            self.logger.debug("Labelled dataset is incomplete and will not be written.")
            raise ValueError()
        