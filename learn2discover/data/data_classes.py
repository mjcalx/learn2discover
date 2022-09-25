from enum import Enum
from typing import List, Optional, Dict

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
