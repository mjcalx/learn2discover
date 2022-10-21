from configs.config_manager import ConfigManager
from data.dataset_manager import DatasetManager
from oracle.stopping_criteria.stopping_criterion import StoppingCriterion


class AnnotationsStoppingCriterion(StoppingCriterion):
    def __init__(self):
        super(AnnotationsStoppingCriterion, self).__init__()
        self.dmgr = DatasetManager.get_instance()
        self.num_annotations = self.settings['num_annotations']

        _unlab_len = len(self.dmgr.data.unlabelled_data)
        if _unlab_len < self.num_annotations:
            _m = 'Not enough unlabelled data for stopping criterion. Needed >={}, received {}.'
            self.logger.error(_m.format(self.num_annotations, _unlab_len))
            raise AssertionError

    @property
    def name(self):
        return 'Annotations Stopping Criterion'
    
    def __call__(self):
        return self.dmgr.data.training_data.count >= self.num_annotations
        
    def report(self):
        self.logger.debug(f'Num Annotations: {self.dmgr.data.training_data.count}')
