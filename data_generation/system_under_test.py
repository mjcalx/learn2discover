import pandas as pd
from loggers.logger_factory import LoggerFactory
from data.data_attributes import DataAttributes
from data.enum import Outcome

class SystemUnderTest:
    def __init__(self, input_attributes, output_attributes, evaluation_attribute, fail_values):
        self.logger = LoggerFactory.get_logger(self._type())
        self.attributes = DataAttributes(input_attributes, output_attributes)
        self.evalutation_attribute = evaluation_attribute
        self.fail_values = fail_values
        assert len(set(output_attributes).intersection(set(input_attributes))) == 0

    def _type(self):
        return self.__class__.__name__

    def evaluate_outcomes(self, outputs: pd.DataFrame) -> pd.Series:
        """
        Determines the outcome for each data instance based on its output values.

        Return a pandas Series of strings of Outcome values.
        """
        assign_outcome = lambda x : Outcome.FAIL.value if str(x[self.evalutation_attribute]) in self.fail_values else Outcome.PASS.value
        self.logger.debug("Evaluating SUT outcomes...")
        return outputs.apply(assign_outcome, axis=1)