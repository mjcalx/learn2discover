import pandas as pd
from system_under_test import SystemUnderTest
from data.enum import Outcome
from data.data_attributes import DataAttributes

class Compas(SystemUnderTest):
    INPUT_ATTRIBUTES = ['Person_ID', 'AssessmentID', 'Case_ID', 'Agency_Text', 'LastName', 'FirstName', 'MiddleName',
                        'Sex_Code_Text', 'Ethnic_Code_Text', 'DateOfBirth', 'ScaleSet_ID', 'ScaleSet', 'AssessmentReason',
                        'Language', 'LegalStatus', 'CustodyStatus', 'MaritalStatus', 'Screening_Date',
                        'RecSupervisionLevel', 'RecSupervisionLevelText', 'Scale_ID']
    OUTPUT_ATTRIBUTES = ['DisplayText', 'RawScore', 'DecileScore', 'ScoreText', 'AssessmentType', 'IsCompleted',
                         'IsDeleted']
    def __init__(self, input_attributes=None, output_attributes=None):
        if input_attributes is None: 
            input_attributes = Compas.INPUT_ATTRIBUTES
        if output_attributes is None:
            output_attributes = Compas.OUTPUT_ATTRIBUTES
        assert len(set(output_attributes).intersection(set(input_attributes))) == 0
        super(Compas, self).__init__(input_attributes, output_attributes)

    def evaluate_outcomes(self, outputs: pd.DataFrame) -> pd.Series:
        """
        Determines the outcome for each data instance based on its output values.

        Return a pandas Series of strings of Outcome values.
        """
        self.logger.debug("Evaluating SUT outcomes...")
        assign_outcome = lambda x : Outcome.FAIL.value if str(x["ScoreText"]) in ["High", "Medium", "nan"] else Outcome.PASS.value
        return outputs.apply(assign_outcome, axis=1)
