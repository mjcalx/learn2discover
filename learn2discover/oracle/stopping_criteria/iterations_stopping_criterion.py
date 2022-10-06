from configs.config_manager import ConfigManager
from oracle.stopping_criteria.stopping_criterion import StoppingCriterion

class IterationsStoppingCriterion(StoppingCriterion):
    def __init__(self):
        super(IterationsStoppingCriterion, self).__init__()
        cfg = ConfigManager.get_instance()
        self.settings = cfg.stopping_criterion_settings
        self.max_iterations = self.settings['num_iterations']
        self.iterations = 0

    @property
    def name(self):
        return 'Iterations Stopping Criterion'
    
    def __call__(self):
        self.iterations += 1
        if self.iterations <= self.max_iterations:
            return False
        return True
        
    def report(self):
        self.logger.debug(f'Iteration: {self.iterations}')