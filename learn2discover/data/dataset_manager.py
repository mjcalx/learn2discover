from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from copy import copy
from itertools import chain

from data.ft_dataframe_dataset import FTDataFrameDataset
from data.ft_tensor_dataset import FTTensorDataset

from loggers.logger_factory import LoggerFactory
from data.schema import Schema, VarType
from data.loader import Loader
from data.enum import Label, Outcome, ParamType
from data.data_attributes import DataAttributes
from utils.validation_utils import ValidationUtils
from utils.logging_utils import Verbosity
from configs.config_manager import ConfigManager
from data.dataset_factory import DatasetFactory
from data.facades import DatasetFacade

class DatasetManager:
    instance = None

    def __init__(self, attributes: DataAttributes=None, seed: int=42):
        """_summary_
        Args:
            random_state (int, optional): random seed. Defaults to 42.
        """
        self.random = np.random.RandomState(seed)
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self._attributes = attributes
        self.config = ConfigManager.get_instance()

        self.loader = Loader()
        self.schema, self._loaded_data = self.loader.load_data()
        self.in_training = self.loader.is_training_mode

        if self.in_training:
            self._data = DatasetFactory.make(self.schema, self._loaded_data, self.random)
            self._attributes = self.data.attributes
            del self._loaded_data

        DatasetManager.instance = self
    
    @staticmethod
    def get_instance() -> DatasetManager:
        if DatasetManager.instance is None:
            DatasetManager.instance = DatasetManager()
        return DatasetManager.instance
    
    @property
    def tensor_data(self) -> TorchDataset | None:
        return self.data if self.in_training else None

    @property
    def data(self) -> FTDataFrameDataset | pd.DataFrame:
        """
        Return a reference to the stored FTDataFrameDataset if in training, or a pandas 
        DataFrame otherwise.
        """
        if self.in_training:
            return self._data
        return self._loaded_data

    @property
    def attributes(self) -> DataAttributes:
        return self._attributes

    def columns_by_type(self, variable_type=None) -> Dict[VarType, list]:
        assert type(variable_type) in [VarType, type(None)]
        self.schema.vars_by_type(variable_type)
        if variable_type is None:
            return self.schema.vars_by_type()
        return self.schema.vars_by_type(variable_type)

    def get_variable_values(self, variable_name: str) -> Tuple[str]:
        return self.schema.get_variable_values(variable_name)

    def get_fairness_testing_dataset(self, outcomes: pd.Series, fairness_labels: pd.Series) -> FTDataFrameDataset:
        """
        Combine and reindex the dataset with SUT outcomes and fairness oracle labels.

        Returns:
            pd.DataFrame: _description_
        """
        if self.in_training:
            return self.data
        
        ValidationUtils.validate_outcomes_series(outcomes)
        ValidationUtils.validate_fairness_labels_series(fairness_labels)

        outcomes.rename(ParamType.OUTCOME.value, inplace=True)
        fairness_labels.rename(ParamType.FAIRNESS.value, inplace=True)

        self.data[ParamType.OUTCOME.value]  = outcomes
        self.data[ParamType.FAIRNESS.value] = fairness_labels

        X = self.attributes.inputs
        Y = self.attributes.outputs
        outcome_idx = ParamType.OUTCOME.value
        fairness_idx = ParamType.FAIRNESS.value

        levels = [
                *[ParamType.INPUTS.value]*len(self.attributes.inputs), 
                *[ParamType.OUTPUTS.value]*len(self.attributes.outputs), 
                ParamType.OUTCOME.value, ParamType.FAIRNESS.value]
        codes = [*self.attributes.inputs, *self.attributes.outputs, 
                outcome_idx, fairness_idx]
        assert len(levels) == len(codes)

        tup = list(zip(*[levels, codes]))
        cols = pd.MultiIndex.from_tuples(tup, names=('param_type', 'param'))
        self.data.columns = cols
        self._ftdata = FTDataFrameDataset(self.schema, self.data)
        return self.data

    def shuffle(self, idxs: pd.Index) -> pd.Index:
        _num_rows = 2
        _show = lambda df : (len(df[:_num_rows:]), idxs[:_num_rows:])
        _m = '{} shuffle(): first {} indices:\n{}'
        self.logger.debug(_m.format('before', *_show(idxs)), verbosity=Verbosity.TALKATIVE)
        _lst_idxs = list(idxs)
        self.random.shuffle(_lst_idxs)
        shuffled = pd.Index(_lst_idxs)
        self.logger.debug(_m.format('after', *_show(shuffled)), verbosity=Verbosity.TALKATIVE)
        return shuffled

    def choose_random_unlabelled(self, unlabelled_idxs: pd.Index) -> pd.Index:
        n = self.config.unlabelled_sampling_size
        return pd.Index(self.random.choice(list(unlabelled_idxs), n, replace=False))
        