from utils.history import History
from oracle.stopping_criteria.stopping_criterion import StoppingCriterion

class ConfidenceStoppingCriterion(StoppingCriterion):
    def __init__(self):
        super(ConfidenceStoppingCriterion, self).__init__()
        self.history = History.get_instance()
        self.conf = self.settings['max_confidence']

    @property
    def name(self):
        return 'Confidence Stopping Criterion'
    
    def __call__(self):
        # Get the current confidence of the model
        return self._get_confidence() >= self.conf
    
    def _get_confidence(self):
        return self.history["Confidence"].iloc[-1]
        
    def report(self):
        self.logger.debug(f'Confidence: {self._get_confidence()} | Threshold: {self.conf}')