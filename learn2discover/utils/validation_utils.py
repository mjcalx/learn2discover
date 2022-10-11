import pandas as pd
from data.enum import Label, Outcome

class ValidationUtils:
    @staticmethod
    def validate_outcomes_series(outcomes: pd.Series) -> None:
        m = '"outcomes" must be a Series of Outcome strings'
        assert all(outcomes.apply(lambda x : x in [o.value for o in Outcome])), m

    @staticmethod
    def validate_fairness_labels_series(fairness_labels: pd.Series) -> None:
        m = '"fairness_labels" must be a Series of Label strings'
        assert all(fairness_labels.apply(lambda x : x in [l.value for l in Label])), m