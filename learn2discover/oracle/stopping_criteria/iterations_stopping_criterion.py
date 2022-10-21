from configs.config_manager import ConfigManager
from data.dataset_manager import DatasetManager
from oracle.stopping_criteria.stopping_criterion import StoppingCriterion


class IterationsStoppingCriterion(StoppingCriterion):
    def __init__(self):
        super(IterationsStoppingCriterion, self).__init__()
        self.max_iterations = self.settings['num_iterations']
        self.iterations = 0

        _min_len = self.max_iterations * ConfigManager.get_instance().unlabelled_sampling_size
        _unlab_len = len(DatasetManager.get_instance().data.unlabelled_data)
        if _min_len > _unlab_len:
            _m = 'Not enough unlabelled data for stopping criterion. Needed >={}, received {}.'
            self.logger.error(_m.format(_min_len, _unlab_len))
            raise AssertionError

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
