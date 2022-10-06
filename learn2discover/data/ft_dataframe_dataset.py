from typing import Callable
from copy import copy
import pandas as pd
import itertools

from data.schema import Schema
from data.data_classes import ParamType, VarType, DataAttributes
from utils.validation_utils import ValidationUtils

class FTDataFrameDataset:
    def __init__(self, schema: Schema, multi_indexed_data: pd.DataFrame):
        #! Assumptions:
        #! - Data is never multiindexed in generation mode
        #! - Data is always multiindexed in training mode
        #! - FTDataFrameDataset is only instantiated in training mode
        assert isinstance(multi_indexed_data.columns, pd.MultiIndex)
        assert isinstance(schema, Schema)
        self.schema = schema
        self.dataset = muilti_indexed_data

        self.attributes = None
        self.attribute_data = None

        # ParamTypes are derived from MultiIndex level 0
        idxs = [self.dataset[x].columns for x in [ParamType.INPUTS.value, ParamType.OUTPUTS.value]]
        attribute_index = pd.Index(list(itertools.chain(*idxs)))
        inputs, outputs = [list(x) for x in idxs]
        self.attributes = DataAttributes(inputs, outputs)
        assert self.attributes is not None

        self.attribute_data = self.dataset.droplevel(0, axis=1)[attribute_index]
        # TODO cast to types

        self._preprocess_categorical_variables()


    # def filter(self, fn: Callable[[pd.DataFrame], pd.DataFrame], **args) -> pd.DataFrame:
    #     """
    #     Function must be applicable to a multiindexed DataFrame
    #     """
    #     return fn(self.dataset)

    @property
    def X(self) -> pd.DataFrame:
        return self.dataset[ParamType.INPUTS.value]
    
    @property
    def Y(self) -> pd.DataFrame:
        return self.dataset[ParamType.OUTPUTS.value]

    @property
    def outcomes(self) -> pd.Series:
        return self.dataset[ParamType.OUTCOME.value][ParamType.OUTCOME.value]

    @property
    def fairness_labels(self) -> pd.Series:
        return self.dataset[ParamType.FAIRNESS.value][ParamType.FAIRNESS.value]
    
    @outcomes.setter
    def outcomes(self, outcomes: pd.Series) -> None:
        ValidationUtils.validate_outcomes_series(outcomes)

        outcomes.rename(ParamType.OUTCOME.value, inplace=True)
        self.dataset[ParamType.OUTCOME.value][ParamType.OUTCOME.value] = outcomes

    @fairness_labels.setter
    def fairness_labels(self, fairness_labels: pd.Series) -> None:
        ValidationUtils.validate_fairness_labels_series(fairness_labels)

        fairness_labels.rename(ParamType.FAIRNESS.value, inplace=True)
        self.dataset[ParamType.FAIRNESS.value][ParamType.FAIRNESS.value] = fairness_labels

    def _preprocess_categorical_variables(self) -> None:
        _cat_cols = self.schema.vars_by_type(VarType.CATEGORICAL)
        for cname in _cat_cols:
            self.attribute_data[cname] = self.attribute_data[cname].astype('category')
            self.dataset.loc(axis=1)[:, cname] = self.dataset.loc(axis=1)[:, cname].astype('category')

