import os
import sys
from configs.config_manager import ConfigManager
from utils.logging_utils import LoggingUtils
from loggers.logger_factory import LoggerFactory

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
        pass
        self.config_manager = ConfigManager.create_instance('.')
        self.config_manager.load_configs()
        LoggingUtils.get_instance().debug('Loaded Configurations.')
        self.logger = LoggerFactory.get_logger(__class__.__name__)
    def run(self):
        pass

if __name__=="__main__":
    main()