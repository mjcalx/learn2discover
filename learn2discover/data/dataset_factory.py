import numpy as np
import pandas as pd
from configs.config_manager import ConfigManager

from data.schema import Schema

from data.ft_dataframe_dataset import FTDataFrameDataset
from data.ft_tensor_dataset import FTTensorDataset
from data.ft_split_dataset import FTSplitDataset
from data.facades import DatasetFacade

class DatasetFactory:
    @staticmethod
    def make(schema: Schema, multi_idx_dataframe: pd.DataFrame, random: np.random.RandomState) -> DatasetFacade :
        cfg = ConfigManager.get_instance()
        ft_data = FTDataFrameDataset(schema, multi_idx_dataframe)
        tensor_data = FTTensorDataset(schema, ft_data)
        split_data = FTSplitDataset(ft_data, random, cfg.test_fraction, cfg.unlabelled_fraction)
        return DatasetFacade(ft_data, tensor_data, split_data)