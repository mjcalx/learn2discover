import os
import sys
from typing import Dict

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
l2d_path = os.path.join(root_path, 'learn2discover')
try:
    sys.path.index(l2d_path)
except ValueError:
    sys.path.append(l2d_path)

from configs.config_manager import ConfigManager
from data.dataset_manager import DatasetManager
from data_generator import DataGenerator
from system_under_test import SystemUnderTest
from oracles.abstract_mock_oracle import AbstractMockOracle

#################Import SUT and Mock Oracle##############
from oracles.group_fairness_oracle import GroupFairnessOracle
#########################################################

if __name__=="__main__":
    """
    A generalised script for generating data for training.
    """
    config  = ConfigManager(os.getcwd(), mode='generate')

    sut    = SystemUnderTest(config.input_attrs, config.output_attrs, config.evaluation_attr, config.fail_vals)
    oracle = GroupFairnessOracle(config.sensitive_attrs)

    datamgr = DatasetManager(sut.attributes)

    data_generator = DataGenerator(sut, oracle)
    dataset = data_generator.generate_data()

    dataset.to_csv(config.output_csv)