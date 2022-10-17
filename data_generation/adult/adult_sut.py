import pandas as pd
from system_under_test import SystemUnderTest
from data.enum import Outcome

class Adult(SystemUnderTest):

    SENSITIVE_ATTRIBUTES =  ["sex", "race"]

    def __init__(self, input_attributes=None, output_attributes=None, oracle_args=None):
        oracle_args = {"sensitive_attributes": Adult.SENSITIVE_ATTRIBUTES}
        super().__init__(input_attributes, output_attributes, oracle_args)
        assert len(set(output_attributes).intersection(set(input_attributes))) == 0

    def evaluate_outcomes(self, outputs: pd.DataFrame) -> pd.Series:
        self.logger.debug("Evaluating SUT outcomes...")
        assign_outcome = lambda x : Outcome.FAIL.value if str(x["salary-prediction"]) == "<=50K" else Outcome.PASS.value
        return outputs.apply(assign_outcome, axis=1)