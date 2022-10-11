import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from loggers.logger_factory import LoggerFactory
from utils.logging_utils import Verbosity
from data.ft_dataframe_dataset import FTDataFrameDataset
from data.enum import ParamType, VarType
from data.schema import Schema

class FTTensorDataset:
    def __init__(self, schema: Schema, ftdata: FTDataFrameDataset):
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self.schema = schema
        self._ftdata = ftdata
        self._tensors = {v:None for v in [VarType.CATEGORICAL, VarType.NUMERICAL]}
        self._labels = torch.tensor(self._ftdata.fairness_labels.cat.codes.values, dtype=torch.int64)
        self._make_tensors()

        col_size = lambda col : len(self._ftdata.flat_index()[col].cat.categories)
        self.categorical_column_sizes = [col_size(col) for col in self.categorical_columns]
        self.logger.debug(f'Column value sizes: {list(zip(self.categorical_columns,self.categorical_column_sizes))}', verbosity=Verbosity.TALKATIVE)
        self.categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in self.categorical_column_sizes]
        self.logger.debug(f'Categorical embedding sizes: {self.categorical_embedding_sizes}', verbosity=Verbosity.TALKATIVE)

    @property
    def tensor_labels(self):
        return self._labels
    @property
    def categorical_columns(self):
        return self._tensors[VarType.CATEGORICAL][0]
    
    @property
    def numerical_columns(self):
        return self._tensors[VarType.NUMERICAL][0]

    def loc(self, idxs: pd.Index) -> Dict[VarType, torch.Tensor]:
        c = VarType.CATEGORICAL
        n = VarType.NUMERICAL

        _get_categorical = lambda idxs : self.get_tensors_of_type[c][1][idxs]
        _get_numerical = lambda idxs : self.get_tensors[n][1][idxs]
        tensors = {
            c:self.get_tensors_of_type(c)[idxs], 
            n:self.get_tensors_of_type(n)[idxs]
        }

        msg = "tensors of size {} (categorical) and {} (numerical) retrieved"
        self.logger.debug(msg.format(tensors[c].size(), tensors[n].size()), verbosity=Verbosity.CHATTY)
        self.logger.debug(f'{tensors[c]}, {tensors[n]}',verbosity=Verbosity.TALKATIVE)

        return tensors
    
    def get_tensors_of_type(self, vartype: VarType) -> torch.Tensor:
        return self._tensors[vartype][1]

    def _make_categorical_tensors(self) -> None:
        """_summary_
        Apply a tensor transformation on the categorical input data and store a
        reference in self._tensors
        """
        df_categorical = self._get_subframe_of_type(VarType.CATEGORICAL)

        category_codes = [df_categorical[i].cat.codes.values for i in df_categorical.columns]
        categorical_data = torch.tensor(np.stack(category_codes, 1), dtype=torch.int64)
        self._tensors[VarType.CATEGORICAL] = (df_categorical.columns, categorical_data)

        self.logger.debug(f'tensor data shape: {self._tensors[VarType.CATEGORICAL][1].size()}')

    def _make_numerical_tensors(self) -> None:
        """_summary_
        Convert input dates to int, concat with numerical inputs and store a 
        reference in self._tensors
        """
        # Get subframe of columns of type DATE and convert to ints
        _date_inputs_subframe = self._get_subframe_of_type(VarType.DATE)
        str_to_int = lambda series : pd.to_datetime(series, infer_datetime_format=True).astype(int)
        df_date = pd.DataFrame({col:str_to_int(_date_inputs_subframe[col]) for col in _date_inputs_subframe.columns})

        df_numerical = self._get_subframe_of_type(VarType.NUMERICAL)

        _combined_subframes = pd.concat([df_numerical, df_date], axis=1)
        _np_data = np.stack([_combined_subframes[col].values for col in _combined_subframes.columns], 1)
        numerical_data = torch.tensor(_np_data, dtype=torch.float)
        self._tensors[VarType.NUMERICAL] = (_combined_subframes.columns, numerical_data)

        self.logger.debug(f'numerical tensor data shape: {self._tensors[VarType.NUMERICAL][1].size()}')

    def _make_tensors(self) -> None:
        self._make_categorical_tensors()
        self._make_numerical_tensors()

    def _get_subframe_of_type(self, vartype: VarType) -> pd.DataFrame:
        _var_cols = self.schema.vars_by_type(vartype)
        input_idxs = _var_cols
        subframe = self._ftdata.flat_index()[input_idxs]
        return subframe
