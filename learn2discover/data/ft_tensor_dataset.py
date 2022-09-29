import torch
import numpy as np
import pandas as pd
from typing import Tuple

from loggers.logger_factory import LoggerFactory
from data.ft_dataframe_dataset import FTDataFrameDataset
from data.data_classes import ParamType, VarType
from data.schema import Schema

class FTTensorDataset:
    def __init__(self, schema: Schema, ftdata: FTDataFrameDataset):
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self.schema = schema
        self._ftdata = ftdata
        _vardict = lambda : {v:None for v in [VarType.CATEGORICAL, VarType.NUMERICAL]}
        self._tensors = {pt:_vardict() for pt in ParamType}  # Dict of VarTypes per ParamType
        self._make_tensors()

    def loc(self, idxs: pd.Index) -> Tuple[torch.Tensor, torch.Tensor]:
        _get_categorical = lambda idxs : self._tensors[ParamType.INPUTS][VarType.CATEGORICAL][idxs]
        _get_numerical = lambda idxs : self._tensors[ParamType.INPUTS][VarType.NUMERICAL][idxs]
        tensors = (_get_categorical(idxs), _get_numerical(idxs))
        msg = "tensors of size {} (categorical) and {} (numerical) retrieved"
        self.logger.debug(msg.format(tensors[0].size(), tensors[1].size()))
        return tensors

    def _make_categorical_tensors(self) -> None:
        """_summary_
        Apply a tensor transformation on the categorical input data and store a
        reference in self._tensors
        """
        categorical_inputs = self._get_subframe_of_type(VarType.CATEGORICAL)

        category_codes = [categorical_inputs[i].cat.codes.values for i in categorical_inputs.columns]
        categorical_data = torch.tensor(np.stack(category_codes, 1), dtype=torch.int64)
        self._tensors[ParamType.INPUTS][VarType.CATEGORICAL] = categorical_data

        self.logger.debug(f'tensor data shape: {self._tensors[ParamType.INPUTS][VarType.CATEGORICAL].shape}')

    def _make_numerical_tensors(self) -> None:
        """_summary_
        Convert input dates to int, concat with numerical inputs and store a 
        reference in self._tensors
        """
        # Get subframe of columns of type DATE and convert to ints
        _date_inputs_subframe = self._get_subframe_of_type(VarType.DATE)
        str_to_int = lambda series : pd.to_datetime(series, infer_datetime_format=True).astype(int)
        date_inputs = pd.DataFrame({col:str_to_int(_date_inputs_subframe[col]) for col in _date_inputs_subframe.columns})

        numerical_inputs = self._get_subframe_of_type(VarType.NUMERICAL)

        _combined_subframes = pd.concat([numerical_inputs, date_inputs], axis=1)
        _np_data = np.stack([_combined_subframes[col].values for col in _combined_subframes.columns], 1)
        numerical_data = torch.tensor(_np_data, dtype=torch.float)
        self._tensors[ParamType.INPUTS][VarType.NUMERICAL] = numerical_data

        self.logger.debug(f'numerical tensor data shape: {self._tensors[ParamType.INPUTS][VarType.NUMERICAL].shape}')

    def _make_tensors(self) -> None:
        self._make_categorical_tensors()
        self._make_numerical_tensors()

    def _get_subframe_of_type(self, vartype: VarType) -> pd.DataFrame:
        _var_cols = self.schema.vars_by_type(vartype)
        input_idxs = [col for col in _var_cols if col in self._ftdata.attributes.inputs]
        subframe = self._ftdata.attribute_data[input_idxs]
        return subframe
