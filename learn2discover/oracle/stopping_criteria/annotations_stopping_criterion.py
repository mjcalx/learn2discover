from configs.config_manager import ConfigManager
from data.dataset_manager import DatasetManager
from oracle.stopping_criteria.stopping_criterion import StoppingCriterion

class AnnotationsStoppingCriterion(StoppingCriterion):
    def __init__(self):
        super(AnnotationsStoppingCriterion, self).__init__()
        cfg = ConfigManager.get_instance()
        self.dmgr = DatasetManager.get_instance()
        self.settings = cfg.stopping_criterion_settings
        self.num_annotations = self.settings['num_annotations']

    @property
    def name(self):
        return 'Annotations Stopping Criterion'
    
    def __call__(self):
        return self.dmgr.data.training_data.count >= self.num_annotations
        
    def report(self):
        self.logger.debug(f'Num Annotations: {self.dmgr.data.training_data.count}')