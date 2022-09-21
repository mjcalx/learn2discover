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


class FairnessLabel(Enum):
    """
    A binary label with values "Fair" and "Unfair"
    """
    FAIR = True
    UNFAIR = False


class TestOutcome(Enum):
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

    def __init__(self, inputs: Dict[str, str], outputs: Dict[str, str], label: Optional[FairnessLabel] = None,
                 outcome: Optional[TestOutcome] = None):
        self._inputs = inputs
        self._outputs = outputs
        self._label: FairnessLabel = label
        self._outcome: TestOutcome = outcome

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label: FairnessLabel):
        self._label = label

    @property
    def outcome(self):
        return self._outcome

    @outcome.setter
    def outcome(self, outcome: TestOutcome):
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


class FileData:
    """
    A class for storing a full file's data
    """

    def __init__(self, attributes: DataAttributes, instances: List[DataInstance]):
        self._attributes = attributes
        self._instances = instances

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, attributes: DataAttributes):
        self._attributes = attributes

    @property
    def instances(self):
        return self._instances

    @instances.setter
    def instances(self, instances: List[DataInstance]):
        self._instances = instances
