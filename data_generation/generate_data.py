import os
import sys
from importlib import util
from os.path import basename, split, splitext
from pathlib import Path

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
l2d_path = os.path.join(root_path, 'learn2discover')
try:
    sys.path.index(l2d_path)
except ValueError:
    sys.path.append(l2d_path)

from configs.config_manager import ConfigManager
from data.dataset_manager import DatasetManager

from data_generator import DataGenerator


def load_module_class(module_name: str, module_path: str):
    module_basename = splitext(basename(module_path))[0]
    _spec = util.spec_from_file_location(module_name, module_path)
    _module = util.module_from_spec(_spec)
    sys.modules[module_name] = _module
    _spec.loader.exec_module(_module)
    _class = getattr(_module, module_name)
    return _class

if __name__=="__main__":
    """
    A generalised script for generating data for training.
    """
    # Get paths for SUT and Mock Oracle
    cfg  = ConfigManager(os.getcwd(), mode='generate')

    # Load the SUT and Mock Oracle
    sut_class = load_module_class(cfg.sut_name, cfg.sut_path)
    oracle_class = load_module_class(cfg.oracle_name, cfg.oracle_path)
    sut = sut_class(cfg.input_attrs, cfg.output_attrs)
    oracle = oracle_class(**sut.oracle_args)

    # Use loaded modules to generate dataset
    datamgr = DatasetManager(sut.attributes)

    data_generator = DataGenerator(sut, oracle)
    dataset = data_generator.generate_data()

    Path(split(cfg.output_csv)[0]).mkdir(parents=True, exist_ok=True)
    dataset.to_csv(cfg.output_csv)
