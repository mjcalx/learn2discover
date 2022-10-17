import os
import sys
from importlib import util

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
l2d_path = os.path.join(root_path, 'learn2discover')
try:
    sys.path.index(l2d_path)
except ValueError:
    sys.path.append(l2d_path)

from configs.config_manager import ConfigManager
from data.dataset_manager import DatasetManager
from data_generator import DataGenerator

if __name__=="__main__":
    """
    A generalised script for generating data for training.
    """
    config  = ConfigManager(os.getcwd(), mode='generate')

    sut_path = config.sut_path
    sut_name = config.sut_name
    sut_module_name = sut_path.split('.')[0].split('/')[-1]

    oracle_path = config.oracle_path
    oracle_name = config.oracle_name
    oracle_module_name = oracle_path.split('.')[0].split('/')[-1]

    # Load the SUT
    sut_spec = util.spec_from_file_location(sut_module_name, sut_path)
    sut_module = util.module_from_spec(sut_spec)
    sys.modules[sut_module_name] = sut_module
    sut_spec.loader.exec_module(sut_module)
    sut_class = getattr(sut_module, sut_name)
    sut = sut_class(config.input_attrs, config.output_attrs)

    # Load the Mock Oracle
    oracle_spec = util.spec_from_file_location(oracle_module_name, oracle_path)
    oracle_module = util.module_from_spec(oracle_spec)
    sys.modules[oracle_module_name] = oracle_module
    oracle_spec.loader.exec_module(oracle_module)
    oracle_class = getattr(oracle_module, oracle_name)
    oracle = oracle_class(**sut.oracle_args)

    datamgr = DatasetManager(sut.attributes)

    data_generator = DataGenerator(sut, oracle)
    dataset = data_generator.generate_data()

    dataset.to_csv(config.output_csv)