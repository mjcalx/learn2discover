import torch
import numpy as np

from loggers.logger_factory import LoggerFactory
from data.ft_dataframe_dataset import FTDataFrameDataset
from data.data_classes import ParamType, VarType
from data.schema import Schema

class FTTensorDataset:
    def __init__(self, schema: Schema, ftdata: FTDataFrameDataset):
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self.schema = schema
        self._ftdata = ftdata
        self._tensors = {pt:dict() for pt in ParamType} # Dict of VarTypes per ParamType

        self._make_categorical_tensors()
    
    def _make_categorical_tensors(self):
        _cat_cols = self.schema.vars_by_type(VarType.CATEGORICAL)
        cat_input_idxs = [col for col in _cat_cols if col in self._ftdata.attributes.inputs]

        cat_inputs = self._ftdata.X[cat_input_idxs]
        cat_codes = [cat_inputs[i].cat.codes.values for i in cat_input_idxs]
        categorical_data = torch.tensor(np.stack(cat_codes, 1), dtype=torch.int64)
        self._tensors[ParamType.INPUTS][VarType.CATEGORICAL] = categorical_data

        self.logger.debug(f'tensor data shape: {self._tensors[ParamType.INPUTS][VarType.CATEGORICAL].shape}')

