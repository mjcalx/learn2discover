from oracle.stopping_criteria.stopping_criterion import StoppingCriterion
from oracle.stopping_criteria.iterations_stopping_criterion import IterationsStoppingCriterion
from oracle.stopping_criteria.annotations_stopping_criterion import AnnotationsStoppingCriterion
from oracle.stopping_criteria.confidence_stopping_criterion import ConfidenceStoppingCriterion
from loggers.logger_factory import LoggerFactory
from utils.logging_utils import Verbosity

class StoppingCriterionFactory:
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self.DEFAULT = IterationsStoppingCriterion

    def get_stopping_criterion(self, type_selection: str) -> StoppingCriterion:
        criterion_types = {
            'iterations'  : IterationsStoppingCriterion,
            'annotations' : AnnotationsStoppingCriterion,
            'confidence'  : ConfidenceStoppingCriterion
        }
        try:
            criterion = criterion_types[type_selection]()
        except KeyError:
            self.logger.debug(f'StoppingCriterion for "{type_selection}" not found. Defaulting to "iterations"')
            criterion = self.DEFAULT()
        finally:
            self.logger.info(f'Select StoppingCriterion for "{type_selection}" : {str(criterion.name)}', verbosity=Verbosity.BASE)
        return criterion