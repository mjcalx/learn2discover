from loggers.logger_factory import LoggerFactory
from configs.config_manager import ConfigManager

class StoppingCriterion:
    def __init__(self):
        self.logger = LoggerFactory.get_logger(self._type())
        cfg = ConfigManager.get_instance()
        self.settings = cfg.stopping_criterion_settings
        super(StoppingCriterion, self).__init__()
    
    def _type(self):
        return self.__class__.__name__

    def __call__(self) -> bool:
        pass

    def report(self) -> None:
        pass
        