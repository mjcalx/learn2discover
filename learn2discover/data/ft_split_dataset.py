import numpy as np
import pandas as pd
from loggers.logger_factory import LoggerFactory
from utils.logging_utils import Verbosity
from configs.config_manager import ConfigManager
from data.enum import ParamType, Label
from data.ft_dataframe_dataset import FTDataFrameDataset

FRACTION_ERR = 0.00001

class SplitFrame:
    def __init__(self, frame: pd.DataFrame):
        """
        Assumes single-indexd columns
        """
        self._frame, self._fair, self._unfair = self._get_data(frame)
        self._count = len(self._frame)
    
    def __len__(self):
        return self._count

    @property
    def index(self):
        return self.all.index
    @property
    def count(self):
        return self._count
    @property
    def all(self):
        return self._frame
    @property
    def fair(self):
        return self._fair
    @property
    def unfair(self):
        return self._unfair

    def _index_fn(self, df: pd.DataFrame, label: Label):
        F = ParamType.FAIRNESS.value
        return df[F][lambda x : x[F] == label].index

    def _get_data(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        all    = df
        fair   = df.loc[self._index_fn(df, Label.FAIR.value)]
        unfair = df.loc[self._index_fn(df, Label.UNFAIR.value)]
        assert isinstance(fair, pd.DataFrame)
        return all, fair, unfair

class FTSplitDataset:
    def __init__(self,
        data: FTDataFrameDataset,
        random_state: np.random.RandomState,
        validation_fraction: float,
        test_fraction: float,
        ):
        '''
        Assumption: `data` does not have a multilevel column index.
        '''
        assert all([ 0 < x < 1 for x in [validation_fraction, test_fraction]])
        # Some of the data need to be unlabelled in order to train the model.
        assert validation_fraction + test_fraction < 1 - FRACTION_ERR

        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self.random = random_state
        n = 'splitting based on test_fraction={}, validation_fraction={}'
        self.logger.debug(n.format(test_fraction, validation_fraction), verbosity=Verbosity.BASE)

        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction

        self.ft_data = data
        self.split = lambda idxs : SplitFrame(self.ft_data.all_columns.loc[idxs])

        self._training_data = None
        self._validation_data = None
        self._test_data = None

        self._valid_test_split()

    def set_training_data(self, idxs: pd.Index) -> None:
        assert self._unlabelled_data is not None
        self._training_data = self.split(idxs)
        overlapping_idxs = self.unlabelled_data.index.intersection(idxs)
        new_unlabelled_idxs = self.unlabelled_data.index.difference(overlapping_idxs)
        self.set_unlabelled_data(new_unlabelled_idxs)
        
    def set_validation_data(self, idxs: pd.Index) -> None:
        self._validation_data = self.split(idxs)

    def set_test_data(self, idxs: pd.Index) -> None:
        self._test_data = self.split(idxs)

    def set_unlabelled_data(self, idxs: pd.Index) -> None:
        self._unlabelled_data = self.ft_data.all_columns.loc[idxs]

    @property
    def training_data(self):
        assert self._training_data is not None
        return self._training_data

    @property
    def validation_data(self):
        assert self._validation_data is not None
        return self._validation_data

    @property
    def test_data(self):
        assert self._test_data is not None
        return self._test_data
    
    @property
    def unlabelled_data(self):
        assert self._unlabelled_data is not None
        return self._unlabelled_data
    
    def _valid_test_split(self) -> None:
        valid_idxs, test_idxs, unlabelled_idxs = self._split_index()

        self.set_validation_data(valid_idxs)
        self.set_test_data(test_idxs)
        self.set_unlabelled_data(unlabelled_idxs)
        self.set_training_data(pd.Index([]))

        self.logger.debug(f'{len(self.ft_data.all_columns)} instances loaded.')

        msg =  'INITIAL SPLIT: \n'
        msg += '                unlabelled={}\n'
        msg += '  training_fair = {} |   training_unfair = {}\n' 
        msg += 'validation_fair = {} | validation_unfair = {}\n'
        msg += '      test_fair = {} |       test_unfair = {}\n'
        
        lengths = [len(d) for d in [
            self.unlabelled_data,
            self.training_data.fair,
            self.training_data.unfair,
            self.validation_data.fair,
            self.validation_data.unfair,
            self.test_data.fair,
            self.test_data.unfair
        ]]
        self.logger.debug(msg.format(*lengths, len(self.unlabelled_data)))

        assert sum(lengths) == len(self.ft_data.all_columns)
        assert sum(lengths) == self.validation_data.count + self.test_data.count + len(self.unlabelled_data)

    def _split_index(self) -> (pd.Index, pd.Index):
        """
        Get indices for each of the validation, test, and unlabelled sets.
        """
        cfg = ConfigManager.get_instance()
        data = self.ft_data.flat_index()

        if cfg.has_human_in_the_loop:
            # Don't simulate unlabelled data if using a human oracle;
            # use actual unlabelled data
            raise NotImplementedError

        test_record_count = int(len(data) * self.test_fraction)
        valid_record_count = max(int(len(data) * self.validation_fraction)-1, 0)
        unlabelled_record_count = len(data) - test_record_count - valid_record_count

        not_in = lambda idxs : ~data.index.isin(idxs)
        
        # Choose test indices
        test_idxs = pd.Index(self.random.choice(data.index, test_record_count, replace=False))
        _idxs_left = data.loc[~data.index.isin(test_idxs)].index

        # Choose validation indices
        valid_idxs = pd.Index(self.random.choice(list(_idxs_left), valid_record_count, replace=False))

        # Use remaining indices for unlabelled set
        if cfg.has_human_in_the_loop:
            unlabelled_idxs = self._get_unlabelled()
        else:
            unlabelled_idxs = data.loc[not_in(test_idxs.union(valid_idxs))].index
            assert len(unlabelled_idxs) == unlabelled_record_count
        
        _m = 'split_dataset(): {} (valid), {} (test), {} (unlabelled), {} (data), {} (total)'
        _lengths = [len(x) for x in [valid_idxs, test_idxs, unlabelled_idxs, data]]
        self.logger.debug(_m.format(*_lengths, sum(_lengths) - len(data)), verbosity=Verbosity.BASE)
        assert len(valid_idxs) + len(test_idxs) + len(unlabelled_idxs) == len(data)
        return valid_idxs, test_idxs, unlabelled_idxs

    def _get_unlabelled(self) -> pd.Index:
        """
        Assumption: unlabelled data were parsed from the same CSV as the rest 
        of the data, with a value of None
        """
        assert self.config.has_human_in_the_loop
        FAIRNESS = ParamType.FAIRNESS.value
        data = self.data.flat_index()

        _label_values = [v.value for v in Label]
        idxs = data[FAIRNESS][~data[FAIRNESS].isin(_label_values)].index
        assert len(idxs) > 0
        self.logger.debug(f'{len(idxs)} unlabelled idxs found', verbosity=Verbosity.BASE)
        return idxs
