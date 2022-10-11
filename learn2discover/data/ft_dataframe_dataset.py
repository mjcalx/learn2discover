from typing import Callable
from copy import copy
import pandas as pd
import itertools

from data.schema import Schema
from data.enum import ParamType, VarType, Label
from data.data_attributes import DataAttributes
from utils.validation_utils import ValidationUtils
from functools import reduce

class FTDataFrameDataset:
    def __init__(self, schema: Schema, multi_indexed_data: pd.DataFrame):
        #! Assumptions:
        #! - Data is never multiindexed in generation mode
        #! - Data is always multiindexed in training mode
        #! - FTDataFrameDataset is only instantiated in training mode
        assert isinstance(multi_indexed_data.columns, pd.MultiIndex)
        assert isinstance(schema, Schema)
        self.schema = schema
        drop_nan_scoretext = lambda df : df.drop(index=df[df.isna()['OUTPUTS']['ScoreText'] == True].index).reset_index(drop=True)
        preprocessors : List[Callable[[pd.DataFrame], pd.DataFrame]] = [drop_nan_scoretext]
        self.dataframe = reduce(lambda f, g: g(f), preprocessors, multi_indexed_data)

        self._flat = None
        self._attributes = None
        # ParamTypes are derived from MultiIndex level 0
        idxs = [self.dataframe[x].columns for x in [ParamType.INPUTS.value, ParamType.OUTPUTS.value]]
        self._attribute_index = pd.Index(list(itertools.chain(*idxs)))
        inputs, outputs = [list(x) for x in idxs]
        self._attributes = DataAttributes(inputs, outputs)
        assert self._attributes is not None
        self._preprocess_categorical_variables()
        self._preprocess_categorical_variables(self.fairness_labels)
        self.flat_index()

    def __len__(self):
        return len(self.dataframe)

    def flat_index(self):
        idxs = list(self._attribute_index) + [ParamType.OUTCOME.value, ParamType.FAIRNESS.value]
        self._flat = self.dataframe.droplevel(0, axis=1)[idxs]
        return self._flat

    def set_attribute_column(self, column_name: str, new_column: pd.Series):
        assert isinstance(new_column, pd.Series), 'column must be of type pandas.Series'
        assert len(new_column) > len(self.dataframe), 'column must be same length as data'
        if column_name in self._categorical_variables():
            self._preprocess_categorical_variables(new_column)
        else:
            self.dataframe.loc(axis=1)[:, column_name] = new_column
    
    @property
    def all_columns(self) -> pd.DataFrame:
        return copy(self.dataframe)

    @property
    def attributes(self) -> DataAttributes:
        return self._attributes

    @property
    def X(self) -> pd.DataFrame:
        return self.dataframe[ParamType.INPUTS.value]
    
    @property
    def Y(self) -> pd.DataFrame:
        return self.dataframe[ParamType.OUTPUTS.value]

    @property
    def outcomes(self) -> pd.Series:
        return self.dataframe[ParamType.OUTCOME.value][ParamType.OUTCOME.value]

    @property
    def fairness_labels(self) -> pd.Series:
        return self.dataframe[ParamType.FAIRNESS.value][ParamType.FAIRNESS.value]
    
    @outcomes.setter
    def outcomes(self, outcomes: pd.Series) -> None:
        ValidationUtils.validate_outcomes_series(outcomes)

        outcomes.rename(ParamType.OUTCOME.value, inplace=True)
        self.dataframe[ParamType.OUTCOME.value,ParamType.OUTCOME.value] = outcomes

    @fairness_labels.setter
    def fairness_labels(self, fairness_labels: pd.Series) -> None:
        ValidationUtils.validate_fairness_labels_series(fairness_labels)

        fairness_labels.rename(ParamType.FAIRNESS.value, inplace=True)
        self.dataframe[ParamType.FAIRNESS.value,ParamType.FAIRNESS.value] = fairness_labels

    def _categorical_variables(self):
        return self.schema.vars_by_type(VarType.CATEGORICAL)


    def _preprocess_categorical_variables(self, series: pd.Series=None) -> None:
        if series is not None:
            assert series.name in self.all_columns.columns
            self.dataframe.loc(axis=1)[:, series.name] = series.astype('category')
            return
        for cname in self._categorical_variables():
            self.dataframe.loc(axis=1)[:, cname] = self.dataframe.loc(axis=1)[:, cname].astype('category')
