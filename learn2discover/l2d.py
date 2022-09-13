import os
import sys
from configs.config_manager import ConfigManager
from utils.logging_utils import LoggingUtils
from loggers.logger_factory import LoggerFactory
from oracle.dataset_manager import DatasetManager
from oracle.query_strategies.query_strategy_factory import QueryStrategyFactory

current_path = os.path.dirname(os.path.realpath(__file__)) 
try:
    sys.path.index(current_path)
except ValueError:
    sys.path.append(current_path)

def main():
    learn_to_discover = Learn2Discover()
    learn_to_discover.run()


class Learn2Discover:
    """
    Learn2Discover
    """

    def __init__(self):
        self.config_manager = ConfigManager.create_instance('.')
        self.config_manager.load_configs()
        LoggingUtils.get_instance().debug('Loaded Configurations.')
        try:
            self.logger = LoggerFactory.get_logger(__class__.__name__)
            self.logger.debug('Loaded Logger.')
            
            self.dataset_manager = DatasetManager()
            self.logger.debug('Loaded DatasetManager.')
            
            self.query_strategies = [QueryStrategyFactory().get_strategy(t) for t in self.config_manager.query_strategies]
        except BaseException as e:
            self.logger.error(f"{e} Exiting...")
            exit()
    def run(self):
        pass

if __name__=="__main__":
    main()