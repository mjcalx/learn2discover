from oracle.stopping_criteria.stopping_criterion import StoppingCriterion
from utils.history import History


class ConfidenceStoppingCriterion(StoppingCriterion):
    def __init__(self):
        super(ConfidenceStoppingCriterion, self).__init__()
        self.history = History.get_instance()
        self.conf = self.settings['max_confidence']
        assert 0 < self.conf <= 1, f"Confidence threshold must be in range (0,1]. Got {self.conf}"

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
