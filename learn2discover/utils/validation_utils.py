import pandas as pd
from data.data_classes import Label, Outcome

class ValidationUtils:
    @staticmethod
    def validate_outcomes_series(outcomes: pd.Series) -> None:
        m = '"outcomes" must be of type Series[Outcome]'
        assert all(outcomes.apply(lambda x : isinstance(x, Outcome))), m

    @staticmethod
    def validate_fairness_labels_series(fairness_labels: pd.Series) -> None:
        m = '"fairness_labels" must be of type Series[Label]'
        assert all(fairness_labels.apply(lambda x : isinstance(x, Label))), m