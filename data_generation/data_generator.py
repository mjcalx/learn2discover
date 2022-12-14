from data.dataset_manager import DatasetManager
from data.ft_dataframe_dataset import FTDataFrameDataset
from loggers.logger_factory import LoggerFactory

from oracles.abstract_mock_oracle import AbstractMockOracle
from suts.abstract_system_under_test import AbstractSystemUnderTest


class DataGenerator:
    def __init__(self, sut: AbstractSystemUnderTest, oracle: AbstractMockOracle):
        """
        This class is a mediator between the DatasetManager, a SystemUnderTest 
        and some AbstractMockOracle.

        This class modifies the state of its DatasetManager to contain the
        results of a fairness oracle's evaluation of the SUT.
        """
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self._dataset_manager = DatasetManager.get_instance()
        self._oracle = oracle
        self._sut = sut

    def generate_data(self) -> FTDataFrameDataset:
        # Call the system under test to evaluate each instance and return an
        # outcome for each to the DatasetManager
        self.logger.debug("Generating data...")
        data = self._dataset_manager.data
        Y = self._sut.attributes.outputs

        outcomes = self._sut.evaluate_outcomes(data[Y])
        fairness_labels = self._oracle.set_labels(outcomes)
        
        self.logger.debug("...done.")
        ftdata = self._dataset_manager.get_fairness_testing_dataset(outcomes, fairness_labels)

        return ftdata
