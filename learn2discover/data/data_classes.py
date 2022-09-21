from enum import Enum
from typing import List, Optional, Dict
from pandas import Series

class DataAttributes:
    """
    Stores information about the data attributes for a dataset
    """

    def __init__(self, inputs: List[str], outputs: List[str]):
        self._inputs = inputs
        self._outputs = outputs

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs


class Label(Enum):
    """
    A binary label with values "Fair" and "Unfair"
    """
    FAIR = True
    UNFAIR = False


class Outcome(Enum):
    """
    A binary outcome with values "Pass" and "Fail"
    """
    PASS = True
    FAIL = False


class DataInstance:
    """
    A class for a single data instance. Should contain:
        - SUT input values
        - SUT output values
        - Fairness label
    """

    def __init__(self, inputs: Dict[str, str], outputs: Dict[str, str], label: Optional[Label] = None,
                 outcome: Optional[Outcome] = None):
        self._inputs = inputs
        self._outputs = outputs
        self._label: Label = label
        self._outcome: Outcome = outcome

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label: Label):
        self._label = label

    @property
    def outcome(self):
        return self._outcome

    @outcome.setter
    def outcome(self, outcome: Outcome):
        self._outcome = outcome

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: List[str]):
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs: List[str]):
        self._outputs = outputs
