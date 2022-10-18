import pandas as pd
from data.enum import Outcome
from suts.abstract_system_under_test import AbstractSystemUnderTest


class German(AbstractSystemUnderTest):

    SENSITIVE_ATTRIBUTES = ["personal_status_and_sex", "is_foreign_worker", "savings_account"]

    def __init__(self, input_attributes=None, output_attributes=None, oracle_args=None):
        oracle_args = {"sensitive_attributes": German.SENSITIVE_ATTRIBUTES}
        super().__init__(input_attributes, output_attributes, oracle_args)
        assert len(set(output_attributes).intersection(set(input_attributes))) == 0

    def evaluate_outcomes(self, outputs: pd.DataFrame) -> pd.Series:
        self.logger.debug("Evaluating SUT outcomes for German Credit...")

        assign_outcome = lambda x: Outcome.PASS.value if x["credit_label"] == 1 else Outcome.FAIL.value
        return outputs.apply(assign_outcome, axis=1)
