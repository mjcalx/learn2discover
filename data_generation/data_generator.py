import pandas as pd
from loggers.logger_factory import LoggerFactory
from data_generation.oracles.abstract_mock_oracle import AbstractMockOracle
from system_under_test import SystemUnderTest
from data.dataset_manager import DatasetManager

class DataGenerator:
    def __init__(self, sut: SystemUnderTest, oracle: AbstractMockOracle):
        """
        This class is a mediator between the DatasetManager, a SystemUnderTest 
        and some MockOracle.

        This class modifies the state of its DatasetManager to contain the
        results of a fairness oracle's evaluation of the SUT.
        """
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self._dataset_manager = DatasetManager.get_instance()
        self._oracle = oracle
        self._sut = sut

    def generate_data(self) -> pd.DataFrame:
        # Call the system under test to evaluate each instance and return an
        # outcome for each to the DatasetManager
        self.logger.debug("Generating data...")
        outcomes = self._sut.evaluate_outcomes(self._dataset_manager.Y)
        self._dataset_manager.set_outcomes(outcomes)

        fairness_labels = self._oracle.set_labels()
        self._dataset_manager.set_fairness_labels(fairness_labels)

        self.logger.debug("...done.")
        return self._dataset_manager.format_dataset()
