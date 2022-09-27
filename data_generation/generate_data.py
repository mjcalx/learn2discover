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
from data_generation.oracles.abstract_mock_oracle import AbstractMockOracle

#################Import SUT and Mock Oracle##############
from compas.compas_sut import Compas
from data_generation.oracles.group_fairness_oracle import GroupFairnessOracle
#########################################################

SUT_TYPE    : SystemUnderTest    = Compas
ORACLE_TYPE : AbstractMockOracle = GroupFairnessOracle
ORACLE_ARGS : Dict               = {'sensitive_attributes': ["Sex_Code_Text", "Ethnic_Code_Text"]}

if __name__=="__main__":
    """
    A generalised script for generating data for training.
    
    Modify SUT_TYPE, ORACLE_TYPE, and ORACLE_ARGS.
    """
    config  = ConfigManager(os.getcwd(), mode='generate')

    sut    = SUT_TYPE()
    oracle = ORACLE_TYPE(**ORACLE_ARGS)

    datamgr = DatasetManager(sut.attributes)

    data_generator = DataGenerator(sut, oracle)
    dataset = data_generator.generate_data()

    dataset.to_csv(config.output_csv)