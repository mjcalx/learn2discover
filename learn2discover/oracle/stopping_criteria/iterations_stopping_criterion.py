from oracle.stopping_criteria.stopping_criterion import StoppingCriterion

class IterationsStoppingCriterion(StoppingCriterion):
    def __init__(self):
        pass

    @property
    def name(self):
        return 'Iterations Stopping Criterion'

    def __call__(self):
        return False
        