from typing import Dict
import pandas as pd
import torch

from data.schema import Schema
from data.data_classes import VarType, DataAttributes
from data.ft_dataframe_dataset import FTDataFrameDataset
from data.ft_tensor_dataset import FTTensorDataset
from data.ft_split_dataset import FTSplitDataset

class FTDataFrameDatasetFacade:
    def __init__(self, ft_dataset: FTDataFrameDataset, **kw):
        self.ft_dataset = ft_dataset
        super(FTDataFrameDatasetFacade, self).__init__(**kw)

    def __len__(self):
        return len(self.ft_dataset)

    @property
    def all_columns(self, multi_indexed=True) -> pd.DataFrame:
        if multi_indexed: 
            return self.ft_dataset.all_columns
        return self.ft_dataset.flat_index()

    def set_attribute_column(self, column_name: str, new_column: pd.Series) -> None:
        self.ft_dataset.set_attribute_column(column_name, new_column)

    @property
    def attributes(self) -> DataAttributes:
        return self.ft_dataset.attributes

    @property
    def X(self) -> pd.DataFrame:
        return self.ft_dataset.X
    
    @property
    def Y(self) -> pd.DataFrame:
        return self.ft_dataset.Y

    @property
    def outcomes(self) -> pd.Series:
        return self.ft_dataset.outcomes()

    @property
    def fairness_labels(self) -> pd.Series:
        return self.ft_dataset.fairness_labels
    
    @outcomes.setter
    def outcomes(self, outcomes: pd.Series) -> None:
        self.ft_dataset.outcomes = outcomes

    @fairness_labels.setter
    def fairness_labels(self, fairness_labels: pd.Series) -> None:
        self.ft_dataset.fairness_labels = fairness_labels

class FTTensorDatasetFacade:
    def __init__(self, tensor_dataset: FTTensorDataset, **kw):
        self.tensor_dataset = tensor_dataset
        super(FTTensorDatasetFacade, self).__init__(**kw)

    @property
    def categorical_embedding_sizes(self):
        return self.tensor_dataset.categorical_embedding_sizes

    @property
    def categorical_columns(self):
        return self.tensor_dataset.categorical_columns

    @property
    def tensor_labels(self):
        return self.tensor_dataset.labels
    
    @property
    def categorical_column_sizes(self):
        return self.tensor_dataset.categorical_column_sizes

    @property
    def categorical_embedding_sizes(self):
        return self.tensor_dataset.categorical_embedding_sizes
    
    @property
    def numerical_columns(self):
        return self.tensor_dataset.numerical_columns

    def loc(self, idxs: pd.Index) -> Dict[VarType, torch.Tensor]:
        return self.tensor_dataset.loc(idxs)
    
    def get_tensors_of_type(self, vartype: VarType) -> torch.Tensor:
        return self.tensor_dataset.get_tensors_of_type(vartype)

class FTSplitDatasetFacade:
    def __init__(self, split_dataset: FTSplitDataset, **kw):
        self.split_dataset = split_dataset
        super(FTSplitDatasetFacade, self).__init__(**kw)

    def set_training_data(self, idxs: pd.Index) -> None:
        self.split_dataset.set_training_data(idxs)

    def set_unlabelled_data(self, idxs: pd.Index) -> None:
        self.split_dataset.set_unlabelled_data(idxs)

    @property
    def training_data(self):
        return self.split_dataset.training_data

    @property
    def validation_data(self):
        return self.split_dataset.validation_data

    @property
    def test_data(self):
        return self.split_dataset.test_data

    @property
    def unlabelled_data(self):
        return self.split_dataset.unlabelled_data
    
class DatasetFacade(FTDataFrameDatasetFacade, FTTensorDatasetFacade, FTSplitDatasetFacade):
    def __init__(self, ft_dataset: FTDataFrameDataset, tensor_dataset: FTTensorDataset, split_dataset: FTSplitDataset):
        super(DatasetFacade, self).__init__(
            ft_dataset=ft_dataset, 
            tensor_dataset=tensor_dataset,
            split_dataset=split_dataset
        )