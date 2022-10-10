from configs.config_manager import ConfigManager
from utils.history import History
from oracle.stopping_criteria.stopping_criterion import StoppingCriterion

class ConfidenceStoppingCriterion(StoppingCriterion):
    def __init__(self):
        super(ConfidenceStoppingCriterion, self).__init__()
        cfg = ConfigManager.get_instance()
        self.history = History.get_instance()
        self.settings = cfg.stopping_criterion_settings
        self.conf = self.settings['max_confidence']

    @property
    def name(self):
        return 'Confidence Stopping Criterion'
    
    def __call__(self):
        # Get the current confidence of the model
        return self.history["Confidence"].iloc[-1] >= self.conf
        
    def report(self):
        self.logger.debug(f'Confidence: {self.history["Confidence"]}')