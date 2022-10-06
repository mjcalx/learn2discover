import numpy as np
import pandas as pd
from loggers.logger_factory import LoggerFactory
from utils.logging_utils import Verbosity
from configs.config_manager import ConfigManager
from data.data_classes import ParamType, Label
from data.ft_dataframe_dataset import FTDataFrameDataset

class FTSplitDataset:
    def __init__(self,
        data: FTDataFrameDataset,
        random_state: np.random.RandomState,
        test_fraction: float,
        unlabelled_fraction: float=0
        ):
        '''
        Assumption: `data` does not have a multilevel column index.
        '''
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self.random = random_state
        n = 'splitting based on test_fraction={}, unlabelled_fraction={}'
        self.logger.debug(n.format(test_fraction, unlabelled_fraction), verbosity=Verbosity.BASE)

        if unlabelled_fraction:
            assert test_fraction + unlabelled_fraction < 1

        self.test_fraction = test_fraction
        self.unlabelled_fraction = unlabelled_fraction

        self.ft_data = data

        self._train_idxs = None
        self._test_idxs = None
        self._unlabelled_idxs = None

        self._training_data = None
        self._evaluation_data = None

        self._training_data_fair = None
        self._training_data_unfair = None

        self._evaluation_data_fair = None
        self._evaluation_data_unfair = None

        self._training_count = None
        self._evaluation_count = None
        self._unlabelled_count = None

        self._train_test_split()

    def set_training_data(self, idxs: pd.Index) -> None:
        self._train_idxs = idxs
        self._training_data, self._training_data_fair, self._training_data_unfair =  self._get_data(idxs)
        self._update_training_count()
        if self._unlabelled_idxs is not None:
            overlapping_idxs = self.unlabelled_idxs.intersection(self._train_idxs)
            new_unlabelled_idxs = self.unlabelled_idxs.difference(overlapping_idxs)
            self.set_unlabelled_data(new_unlabelled_idxs)

    def set_evaluation_data(self, idxs: pd.Index) -> None:
        self._test_idxs = idxs
        self._evaluation_data, self._evaluation_data_fair, self._evaluation_data_unfair = self._get_data(idxs)
        self._update_evaluation_count()
    
    def set_unlabelled_data(self, idxs: pd.Index) -> None:
        self._unlabelled_idxs = idxs
        self._unlabelled_data = self.ft_data.all_columns.loc[idxs]
        self._update_unlabelled_count()

    @property
    def train_idxs(self):
        assert self._train_idxs is not None
        return self._train_idxs

    @property
    def test_idxs(self):
        assert self._test_idxs is not None
        return self._test_idxs

    @property
    def unlabelled_idxs(self):
        assert self._unlabelled_idxs is not None
        return self._unlabelled_idxs

    @property
    def training_data(self):
        assert self._training_data is not None
        return self._training_data

    @property
    def evaluation_data(self):
        assert self._evaluation_data is not None
        return self._evaluation_data

    @property
    def training_data_fair(self):
        assert self._training_data_fair is not None
        return self._training_data_fair

    @property
    def training_data_unfair(self):
        assert self._training_data_unfair is not None
        return self._training_data_unfair

    @property
    def evaluation_data_fair(self):
        assert self._evaluation_data_fair is not None
        return self._evaluation_data_fair
    
    @property
    def evaluation_data_unfair(self):
        assert self._evaluation_data_unfair is not None
        return self._evaluation_data_unfair

    @property
    def unlabelled_data(self):
        assert self._unlabelled_data is not None
        return self._unlabelled_data

    @property
    def training_count(self):
        if self._training_count is None:
            _m = 'SplitDataset()._training_count was referenced before instantiation'
        return self._training_count

    @property
    def evaluation_count(self):
        if self._evaluation_count is None:
            _m = 'SplitDataset()._evaluation_count was referenced before instantiation'
            raise ValueError(_m)
        return self._evaluation_count
    
    @property
    def unlabelled_count(self):
        if self._unlabelled_count is None:
            _m = 'SplitDataset()._evaluation_count was referenced before instantiation'
            raise ValueError(_m)
        return self._unlabelled_count

    def _index_fn(self, index_obj: pd.Index, label: Label):
        F = ParamType.FAIRNESS.value
        return self.ft_data.all_columns.loc[index_obj][F][lambda x : x[F] == label].index

    def _get_data(self, idxs: pd.Index) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        all_idxs    = self.ft_data.all_columns.loc[idxs]
        fair_idxs   = all_idxs.loc[self._index_fn(idxs, Label.FAIR.value)]
        unfair_idxs = all_idxs.loc[self._index_fn(idxs, Label.UNFAIR.value)]
        assert isinstance(fair_idxs, pd.DataFrame)
        return all_idxs, fair_idxs, unfair_idxs

    def _train_test_split(self) -> None:
        train_idxs, test_idxs, unlabelled_idxs = self._split_index()

        self.set_training_data(train_idxs)
        self.set_evaluation_data(test_idxs)
        self.set_unlabelled_data(unlabelled_idxs)

        self.logger.debug(f'{len(self.ft_data.all_columns)} instances loaded.')

        msg = 'NUM_INSTANCES: training_fair={} | training_unfair={} | evaluation_fair={} | evaluation_unfair={} | unlabelled={}'
        lengths = [len(d) for d in [
            self.training_data_fair,
            self.training_data_unfair,
            self.evaluation_data_fair,
            self.evaluation_data_unfair
        ]]
        self.logger.debug(msg.format(*lengths, self.unlabelled_count))

        assert len(self.training_data)==self._training_count
        assert len(self.evaluation_data)==self._evaluation_count
        assert sum(lengths)==len(self.ft_data.all_columns)-self.unlabelled_count==self._training_count+self._evaluation_count

    def _split_index(self) -> (pd.Index, pd.Index):
        cfg = ConfigManager.get_instance()
        data = self.ft_data.flat_index()

        if cfg.has_human_in_the_loop:
            # Don't simulate unlabelled data if using a human oracle;
            # use actual unlabelled data
            self.unlabelled_fraction = 0

        test_record_count = int(len(data) * self.test_fraction)
        unlabelled_record_count = max(int(len(data) * self.unlabelled_fraction)-1, 0)
        train_record_count = len(data) - test_record_count - self.unlabelled_fraction

        not_in = lambda idxs : ~data.index.isin(idxs)
        test_idxs = pd.Index(self.random.choice(data.index, test_record_count, replace=False))
        _idxs_left = data.loc[~data.index.isin(test_idxs)].index

        if cfg.has_human_in_the_loop:
            unlabelled_idxs = self._get_unlabelled()
        else:
            unlabelled_idxs = pd.Index(self.random.choice(list(_idxs_left), unlabelled_record_count, replace=False))
        train_idxs = data.loc[not_in(test_idxs.union(unlabelled_idxs))].index

        _m = 'split_dataset(): {} (train), {} (test), {} (unlabelled), {} (data), {} (total)'
        _lengths = [len(x) for x in [train_idxs, test_idxs, unlabelled_idxs, data]]
        self.logger.debug(_m.format(*_lengths, sum(_lengths) - len(data)), verbosity=Verbosity.BASE)
        assert len(test_idxs) + len(unlabelled_idxs) + len(train_idxs)== len(data)
        return train_idxs, test_idxs, unlabelled_idxs

    def _update_training_count(self):
        assert self._training_data_fair is not None and self._training_data_unfair is not None
        self._training_count = len(self.training_data_fair) + len(self.training_data_unfair)

    def _update_evaluation_count(self):
        assert self.evaluation_data_fair is not None and self.evaluation_data_unfair is not None
        self._evaluation_count = len(self.evaluation_data_fair) + len(self.evaluation_data_unfair)

    def _update_unlabelled_count(self):
        assert self.unlabelled_data is not None
        self._unlabelled_count = len(self.unlabelled_data)
